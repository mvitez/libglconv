--[[ precision-test

Compare precision of hardware and software implementation
run with: qlua precision-network.lua

--]]

require 'nn'
require 'pl'
require 'image'
gl = require 'libglconv'
torch.setdefaulttensortype('torch.FloatTensor')


local iC = 3     -- input channel
local iH = 128   -- input size
local iW = iH
local kC = 4     -- nb kernels
local kH = 3     -- kernel size
local kW = kH
local pH = 2     -- pool size
local pW = pH


-- define network and weights
network = nn.Sequential()
network:add(nn.SpatialConvolutionMM(iC, kC, kH, kW))

for i = 1, kC do
   network.modules[1].weight[i]:fill(0.01*i)
end
network.modules[1].bias:fill(0)

-- use lena as an input
local lena   = image.scale(image.lena(),iW,iH)

-- parse network
dst_hw = torch.Tensor(kC, iH, iW)
filt = torch.Tensor(kC, 3, 3, 3)
for i = 1, kC do
	filt[i]:fill(0.01 * i)
end
local dst_sw = network:forward(lena)
function nn.SpatialConvolutionMM:updateOutput(input)
	if self.weight:dim() == 2 then
		self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
	end
	gl.precision(1)
	gl.logging(1)
	gl.conv(input, self.weight, self.output, self.bias)
	return self.output
end
dst_hw = network:forward(lena)

-- print output
print('==> Precision test')
local precision = 5
local coordinate = 20
local function trunc(x)
   return math.floor(x*math.pow(10,precision)+.5)/math.pow(10,precision)
end

for i = 1, kC do
   local sw = trunc(dst_sw[i][coordinate][coordinate])
   local hw = trunc(dst_hw[i][coordinate][coordinate])
   local diff = trunc(math.abs(sw-hw))
   print('output['..i..']: ', 'CPU = ', sw, 'GPU = ', hw, 'DIFF = ', diff)
end

-- display output
image.display(dst_hw)
