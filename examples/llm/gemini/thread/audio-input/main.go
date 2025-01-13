package main

import (
	"context"
	"fmt"
	"io"
	"os"

	"github.com/henomis/lingoose/llm/gemini"
	"github.com/henomis/lingoose/thread"
	"google.golang.org/genai"
)

var (
	PROJECT = "conversenow-dev"
	REGION  = "us-central1"
)

func main() {
	ctx := context.Background()
	geminiLLM := gemini.New(ctx, gemini.GenerateOpts{
		Project:  PROJECT,
		Location: REGION,
		Model:    gemini.GeminiFlash20Exp,
		Cred:     nil,
		Config:   &genai.GenerateContentConfig{},
	})

	// path to audio file
	file, err := os.Open("examples/llm/gemini/thread/audio-input/spanish_voice_call.wav")
	if err != nil {
		panic(err)
	}
	defer file.Close()
	data, err := io.ReadAll(file)
	if err != nil {
		panic(err)
	}

	systemPrompt := `The Audio is a conversation between two people (Customer and Restaurant) in Spanish. 
Your job is to transcribe the conversation between the two people as a dialog script marking sentences by restaurant as R and customer as C.
Detect the two voices, separate out the conversation between the two voices and provide the transcript of the conversation.
First provide the transcript as is (in the language it is spoken in)
Second provide the transcript translated in english.
`
	t := thread.New().AddMessage(
		thread.NewUserMessage().AddContent(
			thread.NewTextContent(systemPrompt),
		).AddContent(
			thread.NewAudioContent(data, "audio/wav"),
		),
	)

	err = geminiLLM.Generate(context.Background(), t)
	if err != nil {
		panic(err)
	}

	fmt.Println(t.String())
}
