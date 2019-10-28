package core

type (
	Layer interface {
		Init(prev, next Layer)
		Learn(ctxs []Context)
		NewContext() Context
		GetOutputSize() Size
	}

	Context interface {
		Forward(prev Context)
		Backward(next Context)

		GetOutput() Data       // o
		GetErrorToInput() Data // ∂E / ∂a
	}
)
