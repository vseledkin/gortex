package main

import (
	"fmt"
	"log"

	"bufio"
	"os"
	"strings"

	g "github.com/vseledkin/gortex"
)

func main() {
	file := "/volumes/data/64.unique.l.txt"
	dic, e := g.DictionaryFromFile(file, g.WordSplitter{})
	if e != nil {
		panic(e)
	}
	log.Printf("Dic has %d tokens\n", dic.Len())
	dic = dic.Top(1000)
	log.Printf("Dic has %d tokens\n", dic.Len())
	f, e := os.Open(file)
	if e != nil {
		panic(e)
	}
	r := bufio.NewReader(f)
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			break
		}
		line = strings.TrimSpace(line) // remove ending \n
		if len(line) > 0 && dic.AllWordsKnown(line, g.WordSplitter{}) {
			fmt.Printf("%s\n", line)
		}
	}
	f.Close()
}
