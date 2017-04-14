package gortex

import (
	"strings"
	"fmt"
	"unicode"
)

//Tokenizer interface to string tokenizer
type Tokenizer interface {
	Split(str string) []string
}

//WhiteSpaceSplitter space delimited tokens
type WhiteSpaceSplitter struct{}

func (s WhiteSpaceSplitter) Split(str string) []string {
	return strings.Fields(str)
}

//CharSplitter parses string as sequence of characters
type CharSplitter struct{}

func (s CharSplitter) Split(str string) []string {
	runes := []rune(str)
	split := make([]string, len(runes))
	for i, r := range runes {
		split[i] = string(r)
	}
	return split
}

//WordSplitter
type WordSplitter struct{}

func (s WordSplitter) Split(str string) []string {
	var split []string
	token := ""
	for _, r := range str {
		switch {
		case unicode.IsPunct(r) || unicode.IsSymbol(r):
			if len(token) > 0 {
				split = append(split, token)
				token = ""
			}
			split = append(split, string(r))
		case len(token) == 0 && unicode.IsSpace(r):
			continue // skip leading space
		case len(token) == 0 && !unicode.IsSpace(r):
			token = string(r)
		case len(token) > 0 && !unicode.IsSpace(r):
			token += string(r)
		case len(token) > 0 && unicode.IsSpace(r):
			split = append(split, token)
			token = ""
		default:
			panic(fmt.Errorf("unknown symbol %q", r))
		}
	}
	if len(token) > 0 {
		split = append(split, token)
	}
	return split
}