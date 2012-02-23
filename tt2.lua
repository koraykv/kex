require 'kex'

n1=784
n2=784
no=200
m1 = nn.TensorLinear(n1,n2,no)
m2 = nn.TensorLinear(n1,n2,no)
m1.weight:copy(m2.weight)
m1.bias:copy(m2.bias)

in1 = torch.rand(n1)
in2 = torch.rand(n2)
go = torch.rand(no)

m1:updateOutput({in1,in2})
m2:updateOutput2({in1,in2})
print('fdist ', torch.dist(m1.output,m2.output))

m1:zeroGradParameters()
m2:zeroGradParameters()
m1:accGradParameters({in1,in2},go)
m2:accGradParameters2({in1,in2},go)
print('bdist ', torch.dist(m1.gradBias,m2.gradBias))
print('wdist ', torch.dist(m1.gradWeight,m2.gradWeight))


t=torch.tic()
for i=1,10 do
   m1:updateOutput({in1,in2})
   m1:zeroGradParameters()
   m1:updateGradInput({in1,in2},go)
   m1:accGradParameters({in1,in2},go)
   collectgarbage()
end
print('m1 time ', torch.toc(t))

t=torch.tic()
for i=1,10 do
   m2:updateOutput2({in1,in2})
   m2:zeroGradParameters()
   m2:updateGradInput({in1,in2},go)
   m2:accGradParameters2({in1,in2},go)
   collectgarbage()
end
print('m2 time ', torch.toc(t))



