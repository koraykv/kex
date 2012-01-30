local Crop,parent = torch.class('nn.Crop','nn.Module')

local function conv(x,k,s)
   return (x-k)/s + 1
end

local function iconv(x,k,s)
   return (x-1)*s+k
end

local function getSize(m,iw,ih,func)
   local name = torch.typename(m)
   local ow = iw
   local oh = ih
   if name == 'nn.SpatialConvolution' or
      name == 'nn.SpatialConvolutionMap' or
      name == 'nn.SpatialSubSampling' or
      name == 'nn.SpatialMaxPooling' or
      name == 'nn.SpatialLPPooling' then
      ow = func(iw,m.kW,m.dW)
      oh = func(ih,m.kH,m.dH)
   elseif name == 'nn.Sequential' then
      for i=1,#m.modules do
	 ow,oh = getSize(m:get(i),ow,oh,func)
      end
   end
   return ow,oh
end

local function getInputSize(m,ow,oh)
   return getSize(m,ow,oh,iconv)
end

local function getOutputSize(m,iw,ih)
   return getSize(m,iw,ih,conv)
end

function Crop:__init(m,minw,minh)
   parent.__init(self)
   self.module = m
   self.ominw = minw
   self.ominh = minh
   self.iminw,self.iminh = getInputSize(m,self.ominw, self.ominh)
end

function Crop:updateOutput(input)
   local iw = input:size(3)
   local ih = input:size(2)
   if iw < self.iminw or ih < self.iminh then
      error(string.format('too small input iw=%d, ih=%d, minw=%d, mih=%d\n', iw,ih,self.iminw,self.iminh))
   end
   local ow,oh = getOutputSize(self.module,iw,ih)
   if ow ~= math.floor(ow) or oh ~= math.floor(oh) then
      local iiw,iih = getInputSize(self.module,math.floor(ow),math.floor(oh))
      local oo = input:narrow(3,math.floor((iw-iiw)/2)+1,iiw):narrow(2,math.floor((ih-iih)/2)+1,iih)
      self.output:resizeAs(oo):copy(oo)
   else
      self.output:resizeAs(input):copy(input)
   end
   return self.output
end
