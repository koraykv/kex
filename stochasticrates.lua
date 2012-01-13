
function kex.nnhacks()

   local Linear = torch.getmetatable("nn.Linear")
   local oldLinearUpdateParameters = Linear.updateParameters
   function Linear:updateParameters(learningRate)
      -- scale the gradients so that we do not add up bluntly like in batch
      oldLinearUpdateParameters(self, learningRate/self.weight:size(2))
   end
   local oldLinearzeroGradParameters = Linear.zeroGradParameters
   function Linear:zeroGradParameters(momentum)
      if momentum then
	 self.gradWeight:mul(momentum)
	 self.gradBias:mul(momentum)
      else
	 self.gradWeight:zero()
	 self.gradBias:zero()
      end
   end

   local SpatialFullConvolution = torch.getmetatable("nn.SpatialFullConvolution")
   local oldSpatialFullConvolutionUpdateParameters = SpatialFullConvolution.updateParameters
   function SpatialFullConvolution:updateParameters(learningRate)
      oldSpatialFullConvolutionUpdateParameters(self, learningRate/(self.nInputPlane))
   end
   local oldSpatialFullConvolutionZeroGradParameters = SpatialFullConvolution.zeroGradParameters
   function SpatialFullConvolution:zeroGradParameters(momentum)
      if momentum then
	 self.gradWeight:mul(momentum)
      else
	 self.gradWeight:zero()
      end
   end

end
