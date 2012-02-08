
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

mytester:add(nntest)

function kex.module_test()
   mytester:run()
end
