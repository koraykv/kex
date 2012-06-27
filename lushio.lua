
require 'torch'

lushio = {}
lushio5 = {}

function lushio.read(filename)
   -- Reads Lush binary formatted matrix and returns it.
   -- The matrix is stored in 'filename'.
   --
   --   call : x = luahio.readBinaryLushMatrix('my_lush_matrix_file_name');
   --
   -- Inputs:
   --   filename : the name of the lush matrix file. (string)
   --
   -- Outputs:
   --   d   : matrix which is stored in 'filename'.
   --
   --   Koray Kavukcuoglu
   
   local fid = torch.DiskFile(filename,'r'):binary()
   local magic = fid:readInt()
   local ndim = fid:readInt()

   local tdims
   if ndim == 0 then
      tdims = torch.LongStorage({1})
   else
      tdims = fid:readInt(math.max(3,ndim))
   end
   local dims = torch.LongStorage(ndim)
   for i=1,ndim do dims[i] = tdims[i] end

   local nelem = 1
   for i=1,dims:size() do
      nelem = nelem * dims[i]
   end
   local d = torch.Storage()
   local x
   if magic == 507333717 then      --ubyte matrix
      d = fid:readInt(nelem)
      x = torch.ByteTensor(d,1,dims)
   elseif magic == 507333716 then      --integer matrix
      d = fid:readInt(nelem)
      x = torch.IntTensor(d,1,dims)
   elseif magic == 507333713 then      --float matrix
      d = fid:readFloat(nelem)
      x = torch.FloatTensor(d,1,dims)
   elseif magic == 507333715 then      --double matrix
      d = fid:readDouble(nelem)
      x = torch.DoubleTensor(d,1,dims)
   else
      error('Unknown magic number in binary lush matrix')
   end

   fid:close()
   return x
end

function lushio.write(filename,tensor_)
   -- Writes Lush binary formatted matrix.
   -- The tensor is stored in 'filename'.
   --
   --   call : lushio.write('my_lush_matrix_file_name', tensor);
   --
   -- Inputs:
   --   filename : the name of the lush matrix file. (string)
   --   tensor   : torch tensor to be stored
   --
   --   Koray Kavukcuoglu
   local tensor = tensor_:clone()
   local fid = torch.DiskFile(filename,'w'):binary()
   local magic = 0
   if tensor:type() == 'torch.DoubleTensor' then
      magic = 507333715
   elseif tensor:type() == 'torch.FloatTensor' then
      magic = 507333713
   elseif tensor:type() == 'torch.IntTensor' then
      magic = 507333716
   elseif tensor:type() == 'torch.ByteTensor' then
      magic = 507333717
   else
      error('Can not write ' .. tensor:type())
   end
   local ndim = math.max(3,tensor:dim())
   local tdims = torch.IntStorage(ndim)
   tdims:fill(1)
   for i=1,tensor:dim() do tdims[i] = tensor:size(i) end

   fid:writeInt(magic)
   fid:writeInt(tensor:dim())
   fid:writeInt(tdims)
   if magic == 507333717 then      --ubyte matrix
      fid:writeByte(tensor:storage())
   elseif magic == 507333716 then      --integer matrix
      fid:writeInt(tensor:storage())
   elseif magic == 507333713 then      --float matrix 
      fid:writeFloat(tensor:storage())
   elseif magic == 507333715 then      --double matrix
      fid:writeDouble(tensor:storage())
   else
      error('Unknown magic number in tensor.write')
   end
   fid:close()
end

function lushio5.read(filename)
   -- Reads Lush binary formatted matrix and returns it.
   -- The matrix is stored in 'filename'.
   --
   --   call : x = luahio.readBinaryLushMatrix('my_lush_matrix_file_name');
   --
   -- Inputs:
   --   filename : the name of the lush matrix file. (string)
   --
   -- Outputs:
   --   d   : matrix which is stored in 'filename'.
   --
   --   Koray Kavukcuoglu
   
   local fid = torch.DiskFile(filename,'r'):binary()
   local magic = fid:readInt()
   local ndim = fid:readInt()

   local tdims
   if ndim == 0 then
      tdims = torch.LongStorage(1):fill(1)
   else
      tdims = fid:readInt(math.max(3,ndim))
   end
   local dims = torch.LongStorage(tdims:size())
   for i=1,dims:size() do dims[dims:size()-i+1] = tdims[i] end
   --dims:copy(tdims)

   local nelem = 1
   for i=1,dims:size() do
      nelem = nelem * dims[i]
   end
   local d = torch.DoubleStorage()
   local x
   if magic == 507333717 then      --ubyte matrix
      error('ubyte matrices do not exist in Torch')
   elseif magic == 507333716 then      --integer matrix
      d = fid:readInt(nelem)
      x = torch.IntTensor(d,1,dims)
   elseif magic == 507333713 then      --float matrix
      d = fid:readFloat(nelem)
      x = torch.FloatTensor(d,1,dims)
   elseif magic == 507333715 then      --double matrix
      d = fid:readDouble(nelem)
      x = torch.Tensor(d,1,dims)
   else
      error('Unknown magic number in binary lush matrix')
   end

   fid:close()
   return x
end

function lushexport(filename, m, export_output)
   if not m then error('Nil machine') end
   local name = torch.typename(m)
   print(name)
   if export_output and m.output then
      local of = filename .. '_' .. name .. '_output.mat'
      if paths.filep(of) then
	 print('******** this output exists ' .. of)
      end
      lushio.write(of,m.output)
   end
   if name == 'unsup.ConvPSD' then
      lushexport(filename .. '_encoder',m.encoder)
   elseif name == 'nn.SpatialConvolution' then
      local w = m.weight:clone()
      lushio.write(filename .. '_' .. name .. '_convolution_kernel.mat' , w:resize(w:size(1)*w:size(2),w:size(3),w:size(4)))
      local tt = nn.tables.full(m.nInputPlane,m.nOutputPlane)-1
      lushio.write(filename .. '_' .. name .. '_convolution_table.mat' , tt:int())
      lushio.write(filename .. '_' .. name .. '_bias_coeff.mat' , m.bias)
   elseif name == 'nn.SpatialConvolutionMap' then
      lushio.write(filename .. '_' .. name .. '_convolution_kernel.mat' , m.weight)
      lushio.write(filename .. '_' .. name .. '_convolution_table.mat' , m.connTable)
      lushio.write(filename .. '_' .. name .. '_bias_coeff.mat' , m.bias)
   elseif name == 'nn.Diag' then
      lushio.write(filename .. '_' .. name .. '_diag_coeff.mat' , m.weight)
   elseif name == 'nn.DivisiveNormalization' then
      lushio.write(filename .. '_' .. name .. '_kernel.mat', m.kernel)
   elseif name == 'nn.Sequential' then
      for i=1,#m.modules do
	 lushexport(filename .. '_' .. name .. '_layer' .. i, m.modules[i],export_output)
      end
   else
      os.execute('touch ' .. filename .. '_' .. name .. '_noparam.mat')
      print('skipped ' .. name)
   end
end
