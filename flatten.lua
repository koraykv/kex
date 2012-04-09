function kex.flatten(params)
   -- this function flattens arbitrary lists of parameters,
   -- even complex shared ones
   local function flatten(parameters)
      -- already flat ?
      local flat = true
      for k = 2,#parameters do
         if parameters[k]:storage() ~= parameters[k-1]:storage() then
            flat = false
            break
         end
      end
      if flat then
         local nParameters = 0
         for k,param in ipairs(parameters) do
            nParameters = nParameters + param:nElement()
         end
         local flatParameters = parameters[1].new(parameters[1]:storage())
         if nParameters ~= flatParameters:nElement() then
            error('flattenParameters(): weird parameters')
         end
         return flatParameters
      end
      -- compute offsets of each parameter
      local offsets = {}
      local sizes = {}
      local strides = {}
      local elements = {}
      local storageOffsets = {}
      local params = {}
      local nParameters = 0
      for k,param in ipairs(parameters) do
         table.insert(offsets, nParameters+1)
         table.insert(sizes, param:size())
         table.insert(strides, param:stride())
         table.insert(elements, param:nElement())
         table.insert(storageOffsets, param:storageOffset())
         local isView = false
         for i = 1,k-1 do
            if param:storage() == parameters[i]:storage() then
               offsets[k] = offsets[i]
               if storageOffsets[k] ~= storageOffsets[i] or elements[k] ~= elements[i] then
                  error('flattenParameters(): cannot flatten shared weights with different structures')
               end
               isView = true
               break
            end
         end
         if not isView then
            nParameters = nParameters + param:nElement()
         end
      end
      -- create flat vector
      local flatParameters = parameters[1].new(nParameters)
      local storage = flatParameters:storage()
      -- reallocate all parameters in flat vector
      for i = 1,#parameters do
         local data = parameters[i]:clone()
         parameters[i]:set(storage, offsets[i], elements[i]):resize(sizes[i],strides[i]):copy(data)
         data = nil
         collectgarbage()
      end
      -- cleanup
      collectgarbage()
      -- return flat param
      return flatParameters
   end

   local flatParams = {}
   for i=1,#params do
      table.insert(flatParams,flatten(params[i]))
   end

   -- return new flat vector that contains all discrete parameters
   return unpack(flatParams)
end
