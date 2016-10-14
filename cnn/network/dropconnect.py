"""
from keras.layers import Input
from keras.engine.topology import Layer
import scipy as sci
import numpy as np
from scipy.stats import bernoulli
import random


class DropConnect(Layer):

    def __init__(self, input_dim, output_dim, p):
        self.train = True
        self.prob = p or 0.5

        if self.prob >= 1 or self.prob < 0:
            print('[INFO] <LinearDropconnect> illegal percentage, must be 0 <= p < 1')

        # this returns a tensor
        self.noise_weight = Input(shape=(output_dim, input_dim))
        self.noise_bias = Input(shape=output_dim)

        super(DropConnect, self).__init__(type1=input_dim, type2=output_dim)

    def reset(self, stdv):
        if stdv:
            stdv *= sci.sqrt(3)
        else:
            stdv = 1.0 / sci.sqrt(self.weight.size(2))

        if nn.oldSeed:
            for i in xrange(1, self.weight.size(1)):
                self.weight.select(1, i).apply(random.uniform(-stdv, stdv))
                self.bias[i] = random.uniform(-stdv, stdv)
        else:
            self.weight = random.uniform(-stdv, stdv)
            self.bias = random.uniform(-stdv, stdv)

        self.noise_weight.fill(1)
        self.noise_bias.fill(1)

    def updateOutput(input):
        if self.train:
            self.noise_weight = bernoulli(1 - self.prob):cmul(self.weight)
            self.noise_bias = bernoulli(1 - self.prob):cmul(self.bias)

        if input.dim() == 1:
            self.output.resize(self.bias.size(1))
            if self.train:
                self.output.copy(self.noise_bias)
                self.output.addmv(1, self.noise_weight, input)
            else
                self.output.copy(self.bias)
                self.output.addmv(1, self.weight, input)
        elif input.dim() == 2:
            nframe = input.size(1)
            nElement = self.output.nElement()
            self.output.resize(nframe, self.bias:size(1))

            if self.output: nElement() == nElement:
                self.output.zero()

            self.addBuffer = self.addBuffer or input.new()

            if self.addBuffer: nElement() == nframe:
                self.addBuffer:resize(nframe):fill(1)
            if self.train then
                self.output:addmm(0, self.output, 1, input, self.noiseWeight:t())
                self.output:addr(1, self.addBuffer, self.noiseBias)
            else
                self.output:addmm(0, self.output, 1, input, self.weight:t())
                self.output:addr(1, self.addBuffer, self.bias)
        else
            print('[INFO] input must be vector or matrix')

        return self.output

    def updateGradInput(input, grad_output):
        if self.grad_input:
            n_element = self.grad_input.nElement()
            self.grad_input.resizeAs(input)

            if self.grad_input.nElement() == nElement:
                self.grad_input.zero()
            if input.dim() == 1:
                if self.train:
                    self.grad_input.addmv(0, 1, self.noise_weight.t(), grad_output)
                else
                    self.grad_input.addmv(0, 1, self.weight.t(), grad_output)
            elif input.dim() == 2:
                if self.train:
                    self.grad_input.addmm(0, 1, grad_output, self.noise_weight)
                else
                    self.grad_input.addmm(0, 1, grad_output, self.weight)
        return self.grad_input


    def build(self, input_shape):

        self.input_dim = input_shape[1]
        initial_weight_value = np.random.random((self.input_dim, self.output_dim))
        self.W = K.variable(initial_weight_value)
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return input_shape[0], self.output_dim

"""

"""
--[[

   Regularization of Neural Networks using DropConnect
   Li Wan, Matthew Zeiler, Sixin Zhang, Yann LeCun, Rob Fergus

   Dept. of Computer Science, Courant Institute of Mathematical Science, New York University

   Implemented by John-Alexander M. Assael (www.johnassael.com), 2015

]]--

local LinearDropconnect, parent = torch.class('nn.LinearDropconnect', 'nn.Linear')

function LinearDropconnect:__init(inputSize, outputSize, p)

   self.train = true

   self.p = p or 0.5
   if self.p >= 1 or self.p < 0 then
      error('<LinearDropconnect> illegal percentage, must be 0 <= p < 1')
   end

   self.noiseWeight = torch.Tensor(outputSize, inputSize)
   self.noiseBias = torch.Tensor(outputSize)

   parent.__init(self, inputSize, outputSize)
end


function LinearDropconnect:reset(stdv)
   if stdv then
      stdv = stdv * math.sqrt(3)
   else
      stdv = 1./math.sqrt(self.weight:size(2))
   end
   if nn.oldSeed then
      for i=1,self.weight:size(1) do
         self.weight:select(1, i):apply(function()
            return torch.uniform(-stdv, stdv)
         end)
         self.bias[i] = torch.uniform(-stdv, stdv)
      end
   else
      self.weight:uniform(-stdv, stdv)
      self.bias:uniform(-stdv, stdv)
   end

   self.noiseWeight:fill(1)
   self.noiseBias:fill(1)

   return self
end

function LinearDropconnect:updateOutput(input)

   -- Dropconnect
   if self.train then
      self.noiseWeight:bernoulli(1-self.p):cmul(self.weight)
      self.noiseBias:bernoulli(1-self.p):cmul(self.bias)
   end

   if input:dim() == 1 then
      self.output:resize(self.bias:size(1))
      if self.train then
         self.output:copy(self.noiseBias)
         self.output:addmv(1, self.noiseWeight, input)
      else
         self.output:copy(self.bias)
         self.output:addmv(1, self.weight, input)
      end
   elseif input:dim() == 2 then
      local nframe = input:size(1)
      local nElement = self.output:nElement()
      self.output:resize(nframe, self.bias:size(1))
      if self.output:nElement() ~= nElement then
         self.output:zero()
      end
      self.addBuffer = self.addBuffer or input.new()
      if self.addBuffer:nElement() ~= nframe then
         self.addBuffer:resize(nframe):fill(1)
      end
      if self.train then
         self.output:addmm(0, self.output, 1, input, self.noiseWeight:t())
         self.output:addr(1, self.addBuffer, self.noiseBias)
      else
         self.output:addmm(0, self.output, 1, input, self.weight:t())
         self.output:addr(1, self.addBuffer, self.bias)
      end
   else
      error('input must be vector or matrix')
   end

   return self.output
end

function LinearDropconnect:updateGradInput(input, gradOutput)
   if self.gradInput then

      local nElement = self.gradInput:nElement()
      self.gradInput:resizeAs(input)
      if self.gradInput:nElement() ~= nElement then
         self.gradInput:zero()
      end
      if input:dim() == 1 then
         if self.train then
            self.gradInput:addmv(0, 1, self.noiseWeight:t(), gradOutput)
         else
            self.gradInput:addmv(0, 1, self.weight:t(), gradOutput)
         end
      elseif input:dim() == 2 then
         if self.train then
            self.gradInput:addmm(0, 1, gradOutput, self.noiseWeight)
         else
            self.gradInput:addmm(0, 1, gradOutput, self.weight)
         end
      end

      return self.gradInput
   end
end

"""
