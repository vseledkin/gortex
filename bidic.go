package gortex

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
	"strings"
)

type BiDictionary struct {
	First  *Dictionary
	Second *Dictionary
}

func (d BiDictionary) FromFile(file string, s Tokenizer) (*BiDictionary, error) {

	f, e := os.Open(file)
	if e != nil {
		return nil, e
	}
	r := bufio.NewReader(f)
	d.First = &Dictionary{Token2ID: make(map[string]uint), Token2Frequency: make(map[string]uint)}
	d.Second = &Dictionary{Token2ID: make(map[string]uint), Token2Frequency: make(map[string]uint)}
	for {
		line, e := r.ReadString('\n')
		if e != nil {
			break
		}
		line = strings.TrimSpace(line) // remove ending \n
		if len(line) > 0 {
			pair := strings.Split(line, "<->")
			for _, token := range s.Split(pair[0]) {
				d.First.Add(token)
			}
			for _, token := range s.Split(pair[1]) {
				d.Second.Add(token)
			}
		}
	}
	d.First.Add(UNK)
	d.First.Token2Frequency[UNK] = 10e9
	d.First.Add(BOS)
	d.First.Token2Frequency[BOS] = 10e9
	d.First.Add(EOS)
	d.First.Token2Frequency[EOS] = 10e9

	d.Second.Add(UNK)
	d.Second.Token2Frequency[UNK] = 10e9
	d.Second.Add(BOS)
	d.Second.Token2Frequency[BOS] = 10e9
	d.Second.Add(EOS)
	d.Second.Token2Frequency[EOS] = 10e9
	f.Close()
	return &d, nil
}

func (d *BiDictionary) Top(n uint) {
	d.First = d.First.Top(n)
	d.Second = d.Second.Top(n)
}

func (d *BiDictionary) Save(name string) error {
	// save MODEL_NAME
	f, err := os.Create(name)
	if err != nil {
		return err
	}
	encoder := json.NewEncoder(f)
	err = encoder.Encode(d)
	if err != nil {
		return err
	}
	f.Close()
	return nil
}

func (*BiDictionary) Load(name string) (*BiDictionary, error) {
	if len(name) == 0 {
		return nil, fmt.Errorf("no file provided!")
	}
	f, e := os.Open(name)
	if e != nil {
		return nil, e
	}
	var d *BiDictionary
	decoder := json.NewDecoder(f)
	e = decoder.Decode(&d)
	if e != nil {
		return nil, e
	}
	f.Close()
	return d, nil
}
