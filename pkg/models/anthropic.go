package models

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/anthropics/anthropic-sdk-go"
	"github.com/anthropics/anthropic-sdk-go/option"
)

// anthropicDefaultTimeout is used if no timeout is specified
const anthropicDefaultTimeout = 60 * time.Second

// AnthropicModel is a model that uses the Anthropic API.
type AnthropicModel struct {
	Model      string
	ApiKey     string
	MaxTokens  int
	client     *anthropic.Client
	httpClient *http.Client
}

// NewAnthropicModel creates a new AnthropicModel.
func NewAnthropicModel(model string, options ...Option) *AnthropicModel {
	m := &AnthropicModel{
		Model:     model,
		MaxTokens: 1024,
		httpClient: &http.Client{
			Timeout: anthropicDefaultTimeout,
		},
	}

	// Try to get API key from environment variable
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey != "" {
		m.ApiKey = apiKey
	}

	// Apply options
	for _, option := range options {
		option(m)
	}

	// Initialize the Anthropic client with options
	m.client = anthropic.NewClient(
		option.WithAPIKey(m.ApiKey),
		option.WithHTTPClient(m.httpClient),
	)

	return m
}

// Generate generates a response for the given messages.
func (m *AnthropicModel) Generate(ctx context.Context, messages []Message) (string, error) {
	return m.generateInternal(ctx, messages, nil)
}

// GenerateWithTools generates a response for the given messages with tools.
func (m *AnthropicModel) GenerateWithTools(ctx context.Context, messages []Message, tools []map[string]any) (string, error) {
	return m.generateInternal(ctx, messages, tools)
}

// generateInternal is the internal implementation of Generate and GenerateWithTools.
func (m *AnthropicModel) generateInternal(ctx context.Context, messages []Message, tools []map[string]any) (string, error) {
	if m.client == nil {
		return "", errors.New("anthropic client not initialized")
	}

	// Convert our Message type to Anthropic's format
	var systemPrompt string
	var anthropicMessages []anthropic.MessageParam

	for _, msg := range messages {
		// Handle system message separately
		if msg.Role == RoleSystem {
			systemPrompt = msg.Content
			continue
		}

		var anthropicMsg anthropic.MessageParam

		switch msg.Role {
		case RoleUser:
			anthropicMsg = anthropic.NewUserMessage(anthropic.NewTextBlock(msg.Content))
		case RoleAssistant:
			anthropicMsg = anthropic.NewAssistantMessage(anthropic.NewTextBlock(msg.Content))
		case RoleTool:
			// For tool messages, create a text block with the tool result
			// This is a simpler way to handle tool results in messages
			toolMsg := fmt.Sprintf("Tool result from %s: %s", msg.Name, msg.Content)
			anthropicMsg = anthropic.NewAssistantMessage(anthropic.NewTextBlock(toolMsg))
		}

		anthropicMessages = append(anthropicMessages, anthropicMsg)
	}

	// Create the message request params
	params := anthropic.MessageNewParams{
		Model:     anthropic.F(m.Model),
		MaxTokens: anthropic.F(int64(m.MaxTokens)),
		Messages:  anthropic.F(anthropicMessages),
	}

	// Set system prompt if provided
	if systemPrompt != "" {
		// Create a text block for the system prompt
		params.System = anthropic.F([]anthropic.TextBlockParam{
			anthropic.NewTextBlock(systemPrompt),
		})
	}

	// Add tools if provided
	if len(tools) > 0 {
		var anthropicTools []anthropic.ToolParam
		for _, tool := range tools {
			// Extract tool properties
			functionData, ok := tool["function"].(map[string]any)
			if !ok {
				continue
			}

			name, ok := functionData["name"].(string)
			if !ok {
				continue
			}

			// Create tool parameter
			toolParam := anthropic.ToolParam{
				Name: anthropic.F(name),
			}

			// Add description if provided
			if description, ok := functionData["description"].(string); ok {
				toolParam.Description = anthropic.F(description)
			}

			// Add input schema if provided
			if parameters, ok := functionData["parameters"].(map[string]any); ok {
				// Pass parameters map directly as input schema
				// This is a workaround for type system issues
				toolParam.InputSchema = anthropic.F(any(parameters))
			}

			anthropicTools = append(anthropicTools, toolParam)
		}

		if len(anthropicTools) > 0 {
			// This is a workaround for type casting issues
			// Convert the tools with proper type casting
			var toolUnions []anthropic.ToolUnionUnionParam
			for _, tool := range anthropicTools {
				// Convert each tool to the union type
				toolUnions = append(toolUnions, tool)
			}

			// Assign the tools to the params
			params.Tools = anthropic.F(toolUnions)
		}
	}

	// Make the API call
	resp, err := m.client.Messages.New(ctx, params)
	if err != nil {
		return "", fmt.Errorf("failed to create message: %w", err)
	}

	// Process the response
	if len(resp.Content) == 0 {
		return "", errors.New("empty response from model")
	}

	// Check the content type in response
	content := resp.Content[0]

	// Check if there's a tool call
	if content.Type == "tool_use" {
		// For Claude-3, tool_use content has a JSON structure with Name and Input
		// We can access it through Content.JSON which has the raw JSON
		// Extract tool use info from the response
		rawContent, err := json.Marshal(content)
		if err != nil {
			return "", fmt.Errorf("failed to marshal tool_use content: %w", err)
		}

		// Parse the tool use data
		var toolUseData struct {
			ToolUse struct {
				Name  string          `json:"name"`
				Input json.RawMessage `json:"input"`
			} `json:"tool_use"`
		}

		err = json.Unmarshal(rawContent, &toolUseData)
		if err != nil {
			return "", fmt.Errorf("failed to unmarshal tool_use data: %w", err)
		}

		// Create a properly formatted tool call response
		toolResponse := map[string]any{
			"tool": toolUseData.ToolUse.Name,
			"args": toolUseData.ToolUse.Input,
		}

		toolResponseJSON, err := json.Marshal(toolResponse)
		if err != nil {
			return "", err
		}

		return string(toolResponseJSON), nil
	}

	// Return text content
	if content.Type == "text" {
		return content.Text, nil
	}

	return "", errors.New("unsupported response content type")
}

// WithAnthropicModel sets the model for Anthropic API requests.
func WithAnthropicModel(modelName string) Option {
	return func(model any) {
		switch m := model.(type) {
		case *AnthropicModel:
			m.Model = modelName
		}
	}
}

