local TensorLinear,parent = torch.class('nn.TensorLinear','nn.Module')

function TensorLinear:__init(in1,in2,out)
    parent.__init(self)
    self.weight = torch.Tensor(out,in1,in2)
    self.gradWeight = torch.Tensor(out,in1,in2)
    self.bias = torch.Tensor(out)
    self.gradBias = torch.Tensor(out)

    self.gradInput = {torch.Tensor(), torch.Tensor()}
    self:reset()
end

function TensorLinear:reset( stdv )
    if stdv then
        stdv = stdv * math.sqrt(3)
    else
        stdv = 1./math.sqrt(self.weight:size(2)+self.weight:size(3))
    end

    -- we do this so the initialization is exactly
    -- the same than in previous torch versions
    for i=1,self.weight:size(1) do
        -- self.weight:select(1, i):apply(function()
        --                                 return torch.uniform(-stdv, stdv)
        --                                 end)
        self.bias[i] = torch.uniform(-stdv, stdv)
    end
    local tf = torch.FloatTensor(self.weight:size()):uniform(-stdv,stdv)
    self.weight:copy(tf)
    --self.weight:uniform(-stdv,stdv)
end

function TensorLinear:updateOutput1( input )
    local in1,in2 = input[1],input[2]
    local weight = self.weight
    local no = weight:size(1)
    local n1 = in1:size(1)
    local n2 = in2:size(1)
    self.buffer = self.buffer or in1.new():resize(n1,n2)
    local iout = self.buffer:zero()

    iout:addr(in1,in2)
    local i1 = torch.Tensor(iout):resize(n1*n2)
    local w2 = torch.Tensor(weight):resize(no,n1*n2)
    self.output:resize(no)
    self.output:addmv(0,1,w2,i1)
    self.output:add(self.bias)
    return self.output
end

function TensorLinear:updateOutput2( input )
    local in1,in2 = input[1],input[2]
    local weight = self.weight
    local no = weight:size(1)
    local n1 = in1:size(1)
    local n2 = in2:size(1)

    local w2 = torch.Tensor(weight):resize(no*n1,n2)
    self.o2 = self.o2 or torch.Tensor(no*n1)
    local o2 = self.o2
    o2:resize(no*n1)
    o2:addmv(0,1,w2,in2)
    o2:resize(no,n1)
    self.output:resize(no)
    self.output:addmv(0,1,o2,in1)
    self.output:add(self.bias)
    return self.output
end

function TensorLinear:updateGradInput(input, gradOutput)
    local in1,in2 = input[1],input[2]
    local gin1 = self.gradInput[1]
    local gin2 = self.gradInput[2]
    gin1:resizeAs(in1)
    gin2:resizeAs(in2)

    local weight = self.weight
    local no = weight:size(1)
    local n1 = in1:size(1)
    local n2 = in2:size(1)

    local w2 = torch.Tensor(weight):resize(no,n1*n2)
    self.gin = self.gin or torch.Tensor(n1*n2)
    local gin = self.gin
    gin:resize(n1*n2)
    gin:addmv(0,1,w2:t(),gradOutput)
    gin:resize(n1,n2)

    gin1:addmv(0,1,gin,in2)
    gin2:addmv(0,1,gin:t(),in1)
    return self.gradInput
end

function TensorLinear:accGradParameters1( input, gradOutput, scale)
    scale = scale or 1
    local in1,in2 = input[1],input[2]
    local no = self.weight:size(1)
    local n1 = in1:size(1)
    local n2 = in2:size(1)
    local iout = self.buffer
    local i1 = torch.Tensor(iout):resize(in1:size(1)*in2:size(1))

    local gw2 = torch.Tensor(self.gradWeight):resize(no,n1*n2)
    gw2:addr(scale,gradOutput,i1)
    self.gradBias:add(scale, gradOutput)
end

function TensorLinear:accGradParameters2( input, gradOutput, scale)
    scale = scale or 1
    local in1,in2 = input[1],input[2]
    local no = self.weight:size(1)
    local n1 = in1:size(1)
    local n2 = in2:size(1)

    local gw2 = torch.Tensor(self.gradWeight):resize(no*n1,n2)
    self.go1 = self.go1 or torch.Tensor(no,n1)
    local go1 = self.go1
    go1:zero():resize(no,n1)
    go1:addr(gradOutput,in1)
    go1:resize(no*n1)
    gw2:addr(scale,go1,in2)
    self.gradBias:add(scale, gradOutput)
end

TensorLinear.accGradParameters = TensorLinear.accGradParameters2
TensorLinear.updateOutput = TensorLinear.updateOutput2
