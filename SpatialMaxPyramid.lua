local Spm,parent = torch.class('nn.SpatialMaxPyramid','nn.Concat')

function Spm:__init(nLevels)

   parent.__init(self,1)

   for i=1,nLevels do
      local layer = nn.Sequential()
      local nbins = 2^(i-1)
      local maxer = nn.SpatialMaxPooling2(nbins,nbins)
      maxer.fixOutputSize = true
      layer:add(maxer)
      layer:add(nn.Vectorize())
      self:add(layer)
   end
end
