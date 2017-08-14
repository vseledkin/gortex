package gortex

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

func CharSampleVisitor(file string, minLen uint, tokenizer Tokenizer, dictionary *Dictionary, visitor func(epoch int, sample []uint)) error {
	f, e := os.Open(file)
	if e != nil {
		return e
	}
	epoch := 0
	r := bufio.NewReader(f)
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			f.Seek(0, 0)
			line, e = r.ReadString('\n')
			if e != nil {
				break
			}
			epoch++
		}
		line = strings.TrimSpace(line)
		if len(line) > 0 {
			terms := tokenizer.Split(line)
			sample := make([]uint, len(terms))
			for i, term := range terms {
				sample[i] = dictionary.IDByToken(term)
			}
			if len(sample) > int(minLen) {
				visitor(epoch, sample)
			}
		}
	}
	f.Close()
	return nil
}
func CharClassifierSampleVisitor(file string, minLen uint, rewind bool, tokenizer Tokenizer, dictionary *Dictionary, visitor func(sample []uint, label string)) error {
	f, e := os.Open(file)
	if e != nil {
		return e
	}
	r := bufio.NewReader(f)
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			if rewind {
				f.Seek(0, 0)
				line, e = r.ReadString('\n')
				if e != nil {
					break
				}
			} else {
				break
			}
		}
		line = strings.TrimSpace(line)
		if len(line) > 0 {
			index := strings.LastIndex(line, " __label__")
			if index == -1 {
				panic(fmt.Errorf("Wrong classifier sample [%s]", line))
			}
			label := line[index+1:]
			terms := tokenizer.Split(line[:index])
			sample := make([]uint, len(terms))
			for i, term := range terms {
				sample[i] = dictionary.IDByToken(term)
			}
			if len(sample) > int(minLen) {
				visitor(sample, label)
			}
		}
	}
	f.Close()
	return nil
}

func WordSampleVisitor(file string, tokenizer Tokenizer, dictionary *Dictionary, visitor func(epoch int, sample []uint)) error {
	f, e := os.Open(file)
	if e != nil {
		return e
	}
	epoch := 0
	r := bufio.NewReader(f)
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			f.Seek(0, 0)
			line, e = r.ReadString('\n')
			if e != nil {
				break
			}
			epoch++
		}
		//line = strings.TrimSpace(line)
		if len(line) > 0 {
			terms := tokenizer.Split(line)
			sample := make([]uint, len(terms))
			for i, term := range terms {
				sample[i] = dictionary.IDByToken(term)
			}
			visitor(epoch, sample)
		}
	}
	f.Close()
	return nil
}
func WordCharSampleVisitor(file string, tokenizer Tokenizer, dictionary *Dictionary, visitor func(epoch int, sample [][]uint)) error {
	f, e := os.Open(file)
	if e != nil {
		return e
	}
	epoch := 0
	r := bufio.NewReader(f)
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			f.Seek(0, 0)
			//r = bufio.NewReader(f)
			line, e = r.ReadString('\n')
			if e != nil {
				break
			}
			epoch++
		}
		//line = strings.TrimSpace(line)
		if len(line) > 0 {
			terms := tokenizer.Split(line)
			sample := make([][]uint, len(terms))
			for i, term := range terms {
				for _, r := range []rune(term) {
					sample[i] = append(sample[i], dictionary.IDByToken(string(r)))
				}
			}
			visitor(epoch, sample)
		}
	}
	f.Close()
	return nil
}
