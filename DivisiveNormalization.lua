local DivisiveNormalization, parent = torch.class('nn.DivisiveNormalization','nn.Module')

function DivisiveNormalization:__init(nInputPlane, kernel)
   local function gaussian(size,sigma)
      local height = size
      local width = size
      local center_x = width/2 + 0.5
      local center_y = height/2 + 0.5
      
      -- generate kernel
      local gauss = torch.Tensor(height, width)
      for i=1,height do
	 for j=1,width do
	    gauss[i][j] = math.exp(-(math.pow((j-center_x)/(sigma*width),2)/2 + 
				     math.pow((i-center_y)/(sigma*height),2)/2))
	 end
      end
      gauss:div(gauss:sum())
      return gauss
   end

   self.nInputPlane = nInputPlane or 1

   -- KERNEL
   self.kernel = kernel or gaussian(9,1.591/9)
   -- normalize kernel
   self.kernel:div(self.kernel:sum() * self.nInputPlane)
   local padH = math.floor(self.kernel:size(1)/2)
   local padW = math.floor(self.kernel:size(2)/2)

   -- MEAN
   self.meanestimator = nn.Sequential()
   self.meanestimator:add(nn.SpatialZeroPadding(padW, padW, padH, padH))
   self.meanestimator:add(nn.SpatialConvolutionMap(nn.tables.oneToOne(self.nInputPlane),
						   self.kernel:size(2), self.kernel:size(1)))
   self.meanestimator:add(nn.Sum(1))
   self.meanestimator:add(nn.Replicate(self.nInputPlane))
   for i = 1,self.nInputPlane do 
      self.meanestimator.modules[2].weight[i] = self.kernel
   end
   self.meanestimator.modules[2].bias:zero()

   -- STD
   self.stdestimator = nn.Sequential()
   self.stdestimator:add(nn.SpatialZeroPadding(padW, padW, padH, padH))
   self.stdestimator:add(nn.SpatialConvolutionMap(nn.tables.oneToOne(self.nInputPlane),
						   self.kernel:size(2), self.kernel:size(1)))
   self.stdestimator:add(nn.Sum(1))
   self.stdestimator:add(nn.Replicate(self.nInputPlane))
   for i = 1,self.nInputPlane do 
      self.stdestimator.modules[2].weight[i] = self.kernel
   end
   self.stdestimator.modules[2].bias:zero()

   -- other operation
   self.square = nn.Square()
   self.sqrt = nn.Sqrt()
   self.divider = nn.CDivTable()
   self.subtractor = nn.CSubTable()
end

function DivisiveNormalization:updateOutput(input)
   -- mean
   local mean = self.meanestimator:updateOutput(input)
   -- in - mean
   local inzmean = self.subtractor:updateOutput({input,mean})
   -- (in - mean).^2
   local inzmeansq = self.square:updateOutput(inzmean)
   -- sum_j (w_j (in-mean).^2)
   local invar = self.stdestimator:updateOutput(inzmeansq)
   -- sqrt(sum_j (w_j (in-mean).^2))
   local instd = self.sqrt:updateOutput(invar)
   local thres = math.max(instd:mean(),1e-12)
   -- instd(instd<mean(instd)) = mean(instd)
   instd[torch.lt(instd,thres)] = thres
   self.output = self.divider:updateOutput({inzmean,instd})
   return self.output
end

