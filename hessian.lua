
function accDiagHessianParameters(module, input, diagHessianOutput, gw, hw)
   if #gw ~= #hw then
      error('Number of gradients is nto equal to number of hessians')
   end
   module.inputSq = module.inputSq or input.new()
   module.inputSq:resizeAs(input)
   torch.cmul(module.inputSq, input, input)
   -- replace gradients with hessian
   for i=1,#gw do
      local gwname = gw[i]
      local hwname = hw[i]
      local gwval = module[gwname]
      local hwval = module[hwname]
      if hwval == nil then
	 module[hwname] = gwval.new():resizeAs(gwval)
	 hwval = module[hwname]
      end
      module[gwname] = hwval
      module[hwname] = gwval
   end
   module.accGradParameters(module, module.inputSq, diagHessianOutput, 1)
   -- put back gradients
   for i=1,#gw do
      local gwname = gw[i]
      local hwname = hw[i]
      local gwval = module[gwname]
      local hwval = module[hwname]
      module[gwname] = hwval
      module[hwname] = gwval
   end
end

function updateDiagHessianInput(module, input, diagHessianOutput, w, wsq)
   if #w ~= #wsq then
      error('Number of weights is not equal to number of weights squares')
   end
   module.diagHessianInput = module.diagHessianInput or input.new()
   module.diagHessianInput:resizeAs(input)

   local gi = module.gradInput
   module.gradInput = module.diagHessianInput
   for i=1,#w do
      local wname = w[i]
      local wsqname = wsq[i]
      local wval = module[wname]
      local wsqval = module[wsqname]
      if wsqval == nil then
	 module[wsqname] = wval.new()
	 wsqval = module[wsqname]
      end
      wsqval:resizeAs(wval)
      torch.cmul(wsqval, wval, wval)
      module[wsqname] = wval
      module[wname] = wsqval
   end
   module.updateGradInput(module,input,diagHessianOutput)
   for i=1,#w do
      local wname = w[i]
      local wsqname = wsq[i]
      local wval = module[wname]
      local wsqval = module[wsqname]
      module[wname] = wsqval
      module[wsqname] = wval
   end
   module.gradInput = gi
end

function updateDiagHessianInputPointWise(module, input, diagHessianOutput)
   local tdh = diagHessianOutput.new():resizeAs(diagHessianOutput):fill(1)
   updateDiagHessianInput(module,input,tdh,{},{})
   module.diagHessianInput:cmul(module.diagHessianInput)
   module.diagHessianInput:cmul(diagHessianOutput)
end

------------------------------------------------------------------------------------------------------------
-- MODULE
------------------------------------------------------------------------------------------------------------
function nn.Module.updateDiagHessianInput(self, input, diagHessianOutput)
   self.diagHessianInput = self.diagHessianInput or diagHessianOutput
   return self.diagHessianInput
end

function nn.Module.accDiagHessianParameters(self, input, diagHessianOutput)
end


------------------------------------------------------------------------------------------------------------
-- SEQUENTIAL
------------------------------------------------------------------------------------------------------------
function nn.Sequential.initDiagHessianParameters(self)
   for i=1,#self.modules do
      self.modules[i]:initDiagHessianParameters()
   end
end

function nn.Sequential.updateDiagHessianInput(self, input, diagHessianOutput)
   local currentDiagHessianOutput = diagHessianOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentDiagHessianOutput = currentModule:updateDiagHessianInput(previousModule.output, currentDiagHessianOutput)
      currentModule = previousModule
   end
   currentDiagHessianOutput = currentModule:updateDiagHessianInput(input, currentDiagHessianOutput)
   self.diagHessianInput = currentDiagHessianOutput
   return currentDiagHessianOutput
end

function nn.Sequential.accDiagHessianParameters(self, input, diagHessianOutput)
   local currentDiagHessianOutput = diagHessianOutput
   local currentModule = self.modules[#self.modules]
   for i=#self.modules-1,1,-1 do
      local previousModule = self.modules[i]
      currentModule:accDiagHessianParameters(previousModule.output, currentDiagHessianOutput)
      currentDiagHessianOutput = currentModule.diagHessianInput
      currentModule = previousModule
   end
   currentModule:accDiagHessianParameters(input, currentDiagHessianOutput)
end

------------------------------------------------------------------------------------------------------------
-- CRITERION
------------------------------------------------------------------------------------------------------------
function nn.Criterion.updateDiagHessianInput(self, input, diagHessianOutput)
   self.diagHessianInput = self.diagHessianInput or self.output.new()
   return self.diagHessianInput
end

------------------------------------------------------------------------------------------------------------
-- MSECRITERION
------------------------------------------------------------------------------------------------------------
function nn.MSECriterion.updateDiagHessianInput(self, input, diagHessianOutput)
   self.diagHessianInput = self.diagHessianInput or input.new()
   local val = 2
   if self.sizeAverage then
      val = val / input:nElement()
   end
   self.diagHessianInput:resizeAs(input):fill(val)
   return self.diagHessianInput
end

------------------------------------------------------------------------------------------------------------
-- LINEAR
------------------------------------------------------------------------------------------------------------
function nn.Linear.updateDiagHessianInput(self, input, diagHessianOutput)
   updateDiagHessianInput(self, input, diagHessianOutput, {'weight'}, {'weightSq'})
   return self.diagHessianInput
end

function nn.Linear.accDiagHessianParameters(self, input, diagHessianOutput)
   accDiagHessianParameters(self,input, diagHessianOutput, {'gradWeight','gradBias'}, {'diagHessianWeight','diagHessianBias'})
end

function nn.SpatialFullConvolution.updateDiagHessianInput(self, input, diagHessianOutput)
   updateDiagHessianInput(self, input, diagHessianOutput, {'weight'}, {'weightSq'})
   return self.diagHessianInput
end

function nn.SpatialFullConvolution.accDiagHessianParameters(self, input, diagHessianOutput)
   accDiagHessianParameters(self,input, diagHessianOutput, {'gradWeight'}, {'diagHessianWeight'})
end

------------------------------------------------------------------------------------------------------------
-- TANH
------------------------------------------------------------------------------------------------------------
function nn.Tanh.updateDiagHessianInput(self, input, diagHessianOutput)
   updateDiagHessianInputPointWise(self,input, diagHessianOutput)
   return self.diagHessianInput
end

function nn.TanhShrink.updateDiagHessianInput(self, input, diagHessianOutput)
   updateDiagHessianInputPointWise(self.tanh,input, diagHessianOutput)
   self.diagHessianInput = self.diagHessianInput or input.new():resizeAs(input)
   torch.add(self.diagHessianInput, self.tanh.diagHessianInput, diagHessianOutput)
   return self.diagHessianInput
end

function nn.Diag.updateDiagHessianInput(self, input, diagHessianOutput)
   updateDiagHessianInput(self, input, diagHessianOutput, {'weight'}, {'weightSq'})
   return self.diagHessianInput
end

function nn.Diag.accDiagHessianParameters(self, input, diagHessianOutput)
   accDiagHessianParameters(self,input, diagHessianOutput, {'gradWeight'}, {'diagHessianWeight'})
end

