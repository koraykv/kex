----------------------------------------------------------------------
-- hessian.lua: this file appends extra methods to modules in nn,
-- to estimate diagonal elements of the Hessian. This is useful
-- to condition learning rates individually.
----------------------------------------------------------------------
nn.hessian.activate() -- activate Hessian usage

local accDiagHessianParameters = nn.hessian.accDiagHessianParameters
local updateDiagHessianInput = nn.hessian.updateDiagHessianInput
local updateDiagHessianInputPointWise = nn.hessian.updateDiagHessianInputPointWise
local initDiagHessianParameters = nn.hessian.initDiagHessianParameters

----------------------------------------------------------------------
-- SpatialFullConvolution
----------------------------------------------------------------------
function nn.SpatialFullConvolution.updateDiagHessianInput(self, input, diagHessianOutput)
   updateDiagHessianInput(self, input, diagHessianOutput, {'weight'}, {'weightSq'})
   return self.diagHessianInput
end

function nn.SpatialFullConvolution.accDiagHessianParameters(self, input, diagHessianOutput)
   accDiagHessianParameters(self,input, diagHessianOutput, {'gradWeight'}, {'diagHessianWeight'})
end
function nn.SpatialFullConvolution.initDiagHessianParameters(self)
   initDiagHessianParameters(self,{'gradWeight'},{'diagHessianWeight'})
end

----------------------------------------------------------------------
-- SpatialFullConvolutionMap
----------------------------------------------------------------------
function nn.SpatialFullConvolutionMap.updateDiagHessianInput(self, input, diagHessianOutput)
   updateDiagHessianInput(self, input, diagHessianOutput, {'weight'}, {'weightSq'})
   return self.diagHessianInput
end

function nn.SpatialFullConvolutionMap.accDiagHessianParameters(self, input, diagHessianOutput)
   accDiagHessianParameters(self,input, diagHessianOutput, {'gradWeight'}, {'diagHessianWeight'})
end
function nn.SpatialFullConvolutionMap.initDiagHessianParameters(self)
   initDiagHessianParameters(self,{'gradWeight'},{'diagHessianWeight'})
end
