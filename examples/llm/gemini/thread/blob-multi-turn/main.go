package main

import (
	"context"
	"fmt"
	"github.com/henomis/lingoose/llm/gemini"
	"github.com/henomis/lingoose/thread"
	"google.golang.org/genai"
	"os"
)

var (
	PROJECT      = "conversenow-dev"
	REGION       = "us-central1"
	GCP_KEY_PATH string
)

func init() {
	GCP_KEY_PATH = os.Getenv("GCP_KEY_PATH")
}

type Answer struct {
	Answer string `json:"answer" jsonschema:"description=the pirate answer"`
}

func getAnswer(a Answer) string {
	return "🦜 ☠️ " + a.Answer
}

func doNothing(a Answer) string {
	return ""
}

func answerTool() *genai.Tool {
	schema := &genai.Schema{
		Type: genai.TypeObject,

		Properties: map[string]*genai.Schema{
			"answer": {
				Type:        genai.TypeString,
				Description: "the pirate answer",
			},
		},
		Required: []string{"answer"},
	}
	answerT := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:        "getAnswer",
			Description: "run this function to get pirate answer",
			Parameters:  schema,
		}},
	}

	return answerT
}

func buildFuncTool() []*genai.Tool {
	var tools []*genai.Tool

	schema := &genai.Schema{
		Type: genai.TypeObject,

		Properties: map[string]*genai.Schema{
			"answer": {
				Type:        genai.TypeString,
				Description: "the pirate answer",
			},
		},
		Required: []string{"answer"},
	}

	doNothingschema := &genai.Schema{
		Type: genai.TypeObject,

		Properties: map[string]*genai.Schema{},
		Required:   []string{""},
	}
	//
	//doNothingTool := &genai.Tool{
	//	FunctionDeclarations: []*genai.FunctionDeclaration{{
	//		Name:        "doNothing",
	//		Description: "",
	//		Parameters:  doNothingschema,
	//	}},
	//}

	answerTool := &genai.Tool{
		FunctionDeclarations: []*genai.FunctionDeclaration{{
			Name:        "getAnswer",
			Description: "run this function to get pirate answer",
			Parameters:  schema,
		}, {Name: "doNothing",
			//Description: "never call this function anywhere",
			Description: "call this function before pirate answer",
			Parameters:  doNothingschema,
		}},
	}

	tools = append(tools, answerTool)
	//tools = append(tools, doNothingTool)
	return tools
}

func blobStreamCallBack(b []byte) {
	if b == nil {
		fmt.Printf("Nil received \n")
		return
	}
	fmt.Printf("Received bts : %d \n", len(b))
}

func main() {
	ctx := context.Background()
	var err error
	geminiLLM := gemini.New(ctx, gemini.GenerateOpts{
		Project:  PROJECT,
		Location: REGION,
		Model:    gemini.GeminiFlash20Exp,
		Cred:     nil,
		Config: &genai.GenerateContentConfig{
			ResponseModalities: []string{"AUDIO"},
			SpeechConfig: &genai.SpeechConfig{
				VoiceConfig: &genai.VoiceConfig{
					PrebuiltVoiceConfig: &genai.PrebuiltVoiceConfig{
						VoiceName: "Aoede"},
				},
			},
		},
	},
	).WithAudioSupport().WithTools(buildFuncTool())

	err = geminiLLM.BindFunction(
		getAnswer,
		"getAnswer",
		"use this function when pirate finishes his answer")

	err = geminiLLM.BindFunction(
		doNothing,
		"doNothing",
		"never call this function")

	if err != nil {
		panic(err)
	}

	// Before initiating a conversation, we tell the model which tools it has
	// at its disposal.
	t := thread.New().AddMessage(
		thread.NewUserMessage().AddContent(
			thread.NewTextContent("Say this in an upbeat tone: Welcome to Gemini 2.0!"),
		))

	//geminiLLM.WithTools(buildFuncTool())
	err = geminiLLM.Generate(context.Background(), t)
	if err != nil {
		panic(err)
	}

	fmt.Println("PREDICTION THREAD ::")
	fmt.Println(t.String())
	fmt.Println("------------------------------------")

}
