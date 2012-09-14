#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "generic/SpatialFullConvolution.c"
#else

static int nn_(SpatialFullConvolution_updateOutput)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *output = luaT_getfieldcheckudata(L, 1, "output", torch_Tensor);

  luaL_argcheck(L, input->nDimension == 3, 2, "3D tensor expected");

  /* do convolutions */
  THTensor *tweight = THTensor_(newTranspose)(weight,0,1);
  THTensor_(conv2Dmv)(output, 0.0, 1.0, input, tweight, dH, dW, "F", "C");
  THTensor_(free)(tweight);
  
  return 1;
}


static int nn_(SpatialFullConvolution_updateGradInput)(lua_State *L)
{
  /*THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  */
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradInput = luaT_getfieldcheckudata(L, 1, "gradInput", torch_Tensor);
  
  long nOutputPlane = weight->size[1];
  THArgCheck( nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane" );

  /* gradient to input */
  THTensor_(conv2Dmv)(gradInput, 0.0, 1.0, gradOutput, weight, dH, dW, "V", "X");

  return 1;
}

static int nn_(SpatialFullConvolution_accGradParameters)(lua_State *L)
{
  THTensor *input = luaT_checkudata(L, 2, torch_Tensor);  
  THTensor *gradOutput = luaT_checkudata(L, 3, torch_Tensor);
  real scale = luaL_optnumber(L, 4, 1);  
  int dW = luaT_getfieldcheckint(L, 1, "dW");
  int dH = luaT_getfieldcheckint(L, 1, "dH");

  THTensor *weight = luaT_getfieldcheckudata(L, 1, "weight", torch_Tensor);
  THTensor *gradWeight = luaT_getfieldcheckudata(L, 1, "gradWeight", torch_Tensor);
  
  long nOutputPlane = weight->size[1];
  THArgCheck( nOutputPlane == gradOutput->size[0], 1, "Number of output features is not equal to nOutputPlane" );

  /* gradient to kernels */
  THTensor_(conv2DRevger)(gradWeight, 1.0, scale, gradOutput, input, dH, dW);
  return 0;
}

static const struct luaL_Reg nn_(SpatialFullConvolution__) [] = {
  {"SpatialFullConvolution_updateOutput", nn_(SpatialFullConvolution_updateOutput)},
  {"SpatialFullConvolution_updateGradInput", nn_(SpatialFullConvolution_updateGradInput)},
  {"SpatialFullConvolution_accGradParameters", nn_(SpatialFullConvolution_accGradParameters)},
  {NULL, NULL}
};

static void nn_(SpatialFullConvolution_init)(lua_State *L)
{
  luaT_pushmetatable(L, torch_Tensor);
  luaT_registeratname(L, nn_(SpatialFullConvolution__), "nn");
  lua_pop(L,1);
}

#endif
