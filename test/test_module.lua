
local mytester = torch.Tester()
local jac = nn.Jacobian

local precision = 1e-5

local nntest = {}

function nntest.SpatialFullConvolution()
   local from = random.uniform(2,5)
   local to = random.uniform(2,7)
   local ki = random.uniform(2,7)
   local kj = random.uniform(2,7)
   local si = random.uniform(1,3)
   local sj = random.uniform(1,3)
   local ini = random.uniform(10,13)
   local inj = random.uniform(10,13)
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

function nntest.WeightedMSECriterion()
   local from  = random.uniform(100,200)
   local input = torch.Tensor(from):zero()
   local target = lab.randn(from)
   local weight = lab.randn(from)
   local cri = nn.WeightedMSECriterion(weight)
   local module = nn.CriterionModule(cri,target)

   local err = jac.testJacobian(module, input)
   mytester:assertlt(err, precision, 'error on state ')
   
   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.Diag()
   local from = random.uniform(10,20)
   local sz = random.uniform(100,500)
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
   local from = random.uniform(10,20)
   local si = random.uniform(5,10)
   local sj = random.uniform(5,10)
   
   local module = nn.TanhShrink()
   local input = lab.rand(from,si,sj)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision,'error on state')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

function nntest.SPM()
   local from = random.uniform(3,5)
   local si = random.uniform(20,25)
   local sj = random.uniform(20,25)

   local module = nn.SpatialMaxPyramid(3)
   local input = lab.rand(from,si,sj)

   local err = jac.testJacobian(module,input)
   mytester:assertlt(err,precision,'error on state')

   local ferr, berr = jac.testIO(module, input)
   mytester:asserteq(0, ferr, torch.typename(module) .. ' - i/o forward err ')
   mytester:asserteq(0, berr, torch.typename(module) .. ' - i/o backward err ')
end

mytester:add(nntest)

function kex.module_test()
   mytester:run()
end
