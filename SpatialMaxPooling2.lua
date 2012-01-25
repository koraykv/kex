local SpatialMaxPooling2, parent = torch.class('nn.SpatialMaxPooling2', 'nn.Module')

function SpatialMaxPooling2:__init(kW, kH, dW, dH)
   parent.__init(self)

   dW = dW or kW
   dH = dH or kH
   
   self.kW = kW
   self.kH = kH
   self.dW = dW
   self.dH = dH

   self.indices = torch.Tensor()

   self.fixOutputSize = false
end

function SpatialMaxPooling2:updateOutput(input)
   input.nn.SpatialMaxPooling2_updateOutput(self, input)
   return self.output
end

function SpatialMaxPooling2:updateGradInput(input, gradOutput)
   input.nn.SpatialMaxPooling2_updateGradInput(self, input, gradOutput)
   return self.gradInput
end

function SpatialMaxPooling2:empty()
   self.gradInput:resize()
   self.gradInput:storage():resize(0)
   self.output:resize()
   self.output:storage():resize(0)
   self.indices:resize()
   self.indices:storage():resize(0)
end
