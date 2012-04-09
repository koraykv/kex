local CudaTensor = torch.CudaTensor
function CudaTensor:apply(func)
   local t = self:float()
   t:apply(func)
   self:copy(t)
   return self
end

function CudaTensor:max(dim)
   local t = self:float()
   if not dim then
   	return t:max()
   else
   	return t:max(dim)
   end
end
CudaTensor.torch = {}
CudaTensor.torch.uniform = torch.FloatTensor.torch.uniform
CudaTensor.torch.dist = function(a,b) return a:dist(b) end
CudaTensor.torch.rand = function(n) return torch.FloatTensor.torch.rand(n):cuda() end

