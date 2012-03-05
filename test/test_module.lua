
local mytester = torch.Tester()
local jac = nn.Jacobian

local precision = 1e-5

local nntest = {}

function nntest.SpatialFullConvolution()
   local from = torch.uniform(2,5)
   local to = torch.uniform(2,7)
   local ki = torch.uniform(2,7)
   local kj = torch.uniform(2,7)
   local si = torch.uniform(1,3)
   local sj = torch.uniform(1,3)
   local ini = torch.uniform(10,13)
   local inj = torch.uniform(10,13)
   local module = nn.SpatialFullConvolution(from, to, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
   
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialFullConvolutionMap()
   local from = math.ceil(torch.uniform(2,5))
   local to = math.ceil(torch.uniform(2,7))
   local fanin = math.ceil(torch.uniform(1, from))
   local tt = nn.tables.random(from, to, fanin)
   local ki = math.ceil(torch.uniform(2,7))
   local kj = math.ceil(torch.uniform(2,7))
   local si = math.ceil(torch.uniform(1,3))
   local sj = math.ceil(torch.uniform(1,3))
   local ini = math.ceil(torch.uniform(10,13))
   local inj = math.ceil(torch.uniform(10,13))
   local module = nn.SpatialFullConvolutionMap(tt, ki, kj, si, sj)
   local input = torch.Tensor(from, inj, ini):zero()
   
   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
   
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SpatialFullConvolutionCompare()
    from = math.ceil(torch.uniform(2,5))
    to = math.ceil(torch.uniform(2,7))
    tt = nn.tables.full(from, to)
    ki = math.ceil(torch.uniform(2,7))
    kj = math.ceil(torch.uniform(2,7))
    si = math.ceil(torch.uniform(1,3))
    sj = math.ceil(torch.uniform(1,3))
    ini = math.ceil(torch.uniform(10,13))
    inj = math.ceil(torch.uniform(10,13))
    module1 = nn.SpatialFullConvolutionMap(tt, ki, kj, si, sj)
    module2 = nn.SpatialFullConvolution(from, to, ki, kj, si, sj)
    input = torch.rand(from, inj, ini)
    for k=1,tt:size(1) do
       module1.weight[k]:copy(module2.weight[tt[k][1]][tt[k][2]])
    end
   
   local o1 = module1:updateOutput(input)
   local o2 = module2:updateOutput(input)
   mytester:assertlt(o1:dist(o2), precision, 'error on output')

    go1 = torch.rand(o1:size())
    go2 = go1:clone()

   local gi1= module1:updateGradInput(input,go1)
   local gi2 = module2:updateGradInput(input,go2)
   mytester:assertlt(gi1:dist(gi2), precision, 'error on gradInput')

   module1:zeroGradParameters()
   module2:zeroGradParameters()

   module1:accGradParameters(input,go1)
   module2:accGradParameters(input,go2)
   for k=1,tt:size(1) do
      mytester:assertlt(module1.gradWeight[k]:dist(module2.gradWeight[tt[k][1]][tt[k][2]]),precision,'error on gradWeight ' .. k)
   end
end

function nntest.WeightedMSECriterion()
   local from  = torch.uniform(100,200)
   local input = torch.Tensor(from):zero()
   local target = torch.randn(from)
   local weight = torch.randn(from)
   local cri = nn.WeightedMSECriterion(weight)
   local module = nn.CriterionModule(cri,target)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Diag()
   local from = torch.uniform(10,20)
   local sz = torch.uniform(100,500)
   local module = nn.Diag(from)
   local input = torch.Tensor(from,sz)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err, precision, 'error on state')

   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.TanhShrink()
   local from = torch.uniform(10,20)
   local si = torch.uniform(5,10)
   local sj = torch.uniform(5,10)
   
   local module = nn.TanhShrink()
   local input = torch.rand(from,si,sj)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision,'error on state')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SPM()
   local from = torch.uniform(3,5)
   local si = torch.uniform(20,25)
   local sj = torch.uniform(20,25)

   local module = nn.SpatialMaxPyramid(3)
   local input = torch.rand(from,si,sj)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision,'error on state')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.TensorLinear()
   local nf = torch.uniform(2,5)
   local n1 = torch.uniform(10,20)
   local module = nn.TensorLinear(n1,n1,nf)
   local input = torch.Tensor(2,n1):zero()
   local input1 = input[1]
   local input2 = input[2]
   
   local err = jac.testJacobianParameters(module, input, input1, module.gradInput[1])
   mytester:assertlt(err, precision, 'error on state1 ')

   local err = jac.testJacobianParameters(module, input, input2, module.gradInput[2])
   mytester:assertlt(err, precision, 'error on state2 ')
   
   local err = jac.testJacobianParameters(module, input, module.weight, module.gradWeight)
   mytester:assertlt(err , precision, 'error on weight ')

   local err = jac.testJacobianParameters(module, input, module.bias, module.gradBias)
   mytester:assertlt(err , precision, 'error on bias ')
   
   --local ferr, berr = jac.testIO(module, input)
   --mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   --mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end


mytester:add(nntest)

function kex.module_test()
   mytester:run()
end
