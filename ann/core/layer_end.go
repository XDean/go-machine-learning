package core

type EndLayer struct {
	ErrorFunc LossFunc
	Target    Data

	Input        Data
	TotalError   float64 // E
	ErrorToInput Data    // ∂E/∂a
}

func NewEndLayer(errorFunc LossFunc, target Data) *EndLayer {
	return &EndLayer{ErrorFunc: errorFunc, Target: target}
}

func (e *EndLayer) Init(prev, next Layer) {
	// do nothing
}

func (e *EndLayer) Forward(prev Context) {
	e.Input = prev.GetOutput()
	e.TotalError, e.ErrorToInput = e.ErrorFunc.CalcLoss(e.Target, e.Input)
}

func (e *EndLayer) Backward(next Context) {
	// do nothing
}

func (e *EndLayer) Learn(ctxs []Context) {
	// do nothing
}

func (e *EndLayer) NewContext() Context {
	return e
}

func (e *EndLayer) GetOutput() Data {
	return e.Input
}

func (e *EndLayer) GetErrorToInput() Data {
	return e.ErrorToInput
}

func (e *EndLayer) GetOutputSize() Size {
	return e.Input.Size
}

func (e *EndLayer) ToResult() Result {
	return Result{
		Output:     e.Input,
		Target:     e.Target,
		TotalError: e.TotalError,
	}
}
