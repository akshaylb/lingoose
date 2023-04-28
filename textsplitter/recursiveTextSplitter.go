package textsplitter

import (
	"strings"

	"github.com/henomis/lingoose/document"
)

type recursiveCharacterTextSplitter struct {
	textSplitter
	separators []string
}

func NewRecursiveCharacterTextSplitter(chunkSize int, chunkOverlap int, separators []string, lengthFunction LenFunction) *recursiveCharacterTextSplitter {

	if lengthFunction == nil {
		lengthFunction = func(s string) int {
			return len(s)
		}
	}

	if len(separators) == 0 {
		separators = []string{"\n\n", "\n", " ", ""}
	}

	return &recursiveCharacterTextSplitter{
		textSplitter: textSplitter{
			chunkSize:      chunkSize,
			chunkOverlap:   chunkOverlap,
			lengthFunction: lengthFunction,
		},
		separators: separators,
	}

}

// AI-translated from https://github.com/hwchase17/langchain/blob/master/langchain/text_splitter.py

func (r *recursiveCharacterTextSplitter) SplitDocuments(documents []document.Document) []document.Document {

	docs := make([]document.Document, 0)

	for i, doc := range documents {
		for _, chunk := range r.SplitText(doc.Content) {

			metadata := make(map[string]interface{})
			for k, v := range documents[i].Metadata {
				metadata[k] = v
			}

			docs = append(docs,
				document.Document{
					Content:  chunk,
					Metadata: metadata,
				},
			)
		}
	}

	return docs
}

func (r *recursiveCharacterTextSplitter) SplitText(text string) []string {
	// Split incoming text and return chunks.
	finalChunks := []string{}
	// Get appropriate separator to use
	separator := r.separators[len(r.separators)-1]
	for _, s := range r.separators {
		if s == "" {
			separator = s
			break
		}
		if strings.Contains(text, s) {
			separator = s
			break
		}
	}
	// Now that we have the separator, split the text
	var splits []string
	if separator != "" {
		splits = strings.Split(text, separator)
	} else {
		splits = strings.Split(text, "")
	}
	// Now go merging things, recursively splitting longer texts.
	goodSplits := []string{}
	for _, s := range splits {
		if r.lengthFunction(s) < r.chunkSize {
			goodSplits = append(goodSplits, s)
		} else {
			if len(goodSplits) > 0 {
				mergedText := r.mergeSplits(goodSplits, separator)
				finalChunks = append(finalChunks, mergedText...)
				goodSplits = []string{}
			}
			otherInfo := r.SplitText(s)
			finalChunks = append(finalChunks, otherInfo...)
		}
	}
	if len(goodSplits) > 0 {
		mergedText := r.mergeSplits(goodSplits, separator)
		finalChunks = append(finalChunks, mergedText...)
	}
	return finalChunks
}