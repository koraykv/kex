#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/util.c"
#else

static int kex_(shrinkage)(lua_State *L)
{
  real lambda = luaL_checknumber(L,2);
  THTensor *tensor = luaT_checkudata(L,1, torch_Tensor);
  luaL_argcheck(L, lambda >=0, 2, "Lambda should be non-negative");

  if (lambda == 0) return 1;

  TH_TENSOR_APPLY(real, tensor,
		  if (*tensor_data > lambda)
		  {
		    *tensor_data -= lambda;
		  }
		  else if (*tensor_data < -lambda)
		  {
		    *tensor_data += lambda;
		  }
		  else
		  {
		    *tensor_data = 0;
		  });
  return 1;
}

static int kex_(sign)(lua_State *L)
{
  int narg = lua_gettop(L);
  THTensor *tensor = NULL;
  THTensor *r = luaT_checkudata(L,1,torch_Tensor);
  if (narg == 1)
  {
    tensor = r;
  }
  else if (narg == 2)
  {
    tensor = luaT_checkudata(L,2,torch_Tensor);
  }
  else
  {
    luaL_error(L,"1 or 2 input tensors expected");
  }
  TH_TENSOR_APPLY2(real, r, real, tensor,
		   if (*tensor_data > 0)
		     *r_data = 1;
		   else if (*tensor_data < 0)
		     *r_data = -1;
		   else
		     *r_data = 0;);
  return 1;
}

static const struct luaL_Reg kex_(util__) [] = {
  {"shrinkage", kex_(shrinkage)},
  {"sign", kex_(sign)},
  {NULL, NULL}
};

static void kex_(util_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaL_register(L, NULL, kex_(util__));
  lua_pop(L,1);
}

#endif
