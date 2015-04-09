require 'nn'
require 'sys'

gl = require 'libglconv'
gl.logging(1)
gl.useintp(1)
--gl.precision(1)	-- required on NVidia
torch.setdefaulttensortype('torch.FloatTensor')

local iC = 128    -- input channel
local iH = 128   -- input size
local iW = iH
local kC = 128     -- nb kernels
local kH = 3     -- kernel size
local kW = kH
local pH = 16     -- pool size
local pW = pH

if  arg[1] == nil then
	print('Syntax: th stest.lua <gpu>/<cpu>')
	os.exit(0)
end

print('Initializing')
input = torch.randn(pH, iC, iH, iW)

-- define network and weights
network = nn.Sequential()
network:add(nn.SpatialConvolutionMM(iC, kC, kH, kW))

if arg[1] == 'cpu' then
	print('Running with CPU')
	sys.tic()
	local dst_sw = network:forward(input)
	print(sys.toc()..' seconds')
end

function nn.SpatialConvolutionMM:updateOutput(input)
	gl.precision(1)
	if self.weight:dim() == 2 then
		self.weight = self.weight:view(self.nOutputPlane, self.nInputPlane, self.kH, self.kW)
	end
	gl.conv(input, self.weight, self.output, self.bias)
	return self.output
end

if arg[1] == 'gpu' then
	print('Running with GPU')
	sys.tic()
	dst_sw = network:forward(input)
	print(sys.toc()..' seconds')
end
