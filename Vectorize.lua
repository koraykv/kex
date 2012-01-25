local Vectorize,parent = torch.class('nn.Vectorize','nn.Module')

function Vectorize:__init()
   parent.__init(self)
end

function Vectorize:updateOutput(input)
   local output = input:contiguous()
   self.output:set(output):resize(output:nElement())
   return self.output
end

function Vectorize:updateGradInput(input, gradOutput)
   local gradInput = gradOutput:contiguous()
   self.gradInput:set(gradInput):resize(input:size())
   return self.gradInput
end
