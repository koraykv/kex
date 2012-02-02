local Vectorize,parent = torch.class('nn.Vectorize','nn.Module')

function Vectorize:__init()
   parent.__init(self)
end

function Vectorize:updateOutput(input)
   self.output:resize(input:nElement()):copy(input)
   return self.output
end

function Vectorize:updateGradInput(input, gradOutput)
   self.gradInput:resize(input:size()):copy(gradOutput)
   return self.gradInput
end
