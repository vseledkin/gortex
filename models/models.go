package models

import g "github.com/vseledkin/gortex"

type Classifier interface {
	Forward(G *g.Graph, tokens []string) (class uint, confidence float32)
	Load(modelFile string) error
	Save(modelFile string) error
	GetClasses() map[string]*struct{ Index, Count uint }
	GetClassName(c uint) string
	GetLogits() *g.Matrix
	GetRepresentation() *g.Matrix
	GetParameters() map[string]*g.Matrix
	GetActiveParameters() map[string]*g.Matrix
	GetName() string
	Attention() (map[int]float32, map[int]string)
}
