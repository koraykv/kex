local SqrtBias, parent = torch.class('nn.SqrtBias','nn.Module')

function SqrtBias:__init(b)
   parent.__init(self)
   self.bias = b or 0.01
end

function SqrtBias:updateOutput(input)
   return input.nn.SqrtBias_updateOutput(self,input)
end

function SqrtBias:updateGradInput(input, gradOutput)
   return input.nn.SqrtBias_updateGradInput(self,input,gradOutput)
end
