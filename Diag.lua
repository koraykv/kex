local Diag,parent = torch.class('nn.Diag','nn.Module')

function Diag:__init(nFeature)
   parent.__init(self)
   self.weight = torch.Tensor(nFeature)
   self.gradWeight = torch.Tensor(nFeature)

   self:reset()
end

function Diag:reset(stdv)
   self.weight:fill(1)
end

function Diag:updateOutput(input)
   self.output:resizeAs(input):copy(input)
   for i=1,input:size(1) do
      self.output:select(1,i):mul(self.weight[i])
   end
   return self.output
end

function Diag:updateGradInput(input, gradOutput)
   self.gradInput:resizeAs(gradOutput):copy(gradOutput)
   for i=1,input:size(1) do
      self.gradInput:select(1,i):mul(self.weight[i])
   end
   return self.gradInput
end

function Diag:accGradParameters(input, gradOutput)
   for i=1,input:size(1) do
      self.gradWeight[i] = self.gradWeight[i] + gradOutput:select(1,i):dot(input:select(1,i))
   end
end
