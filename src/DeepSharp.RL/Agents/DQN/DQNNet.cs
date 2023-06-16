﻿namespace DeepSharp.RL.Agents
{
    /// <summary>
    ///     DQN Dep Model
    /// </summary>
    public sealed class DQNNet : Module<torch.Tensor, torch.Tensor>
    {
        private readonly Module<torch.Tensor, torch.Tensor> conv;
        private readonly Module<torch.Tensor, torch.Tensor> fc;

        public DQNNet(long[] inputShape, int actions, DeviceType deviceType = DeviceType.CUDA) :
            base("DQNN")
        {
            var modules = new List<(string, Module<torch.Tensor, torch.Tensor>)>
            {
                ("Conv2d1", Conv2d(inputShape[0], 32, 8, 4)),
                ("Relu1", ReLU()),
                ("Conv2d2", Conv2d(32, 64, 4, 2)),
                ("Relu2", ReLU()),
                ("Conv2d3", Conv2d(64, 64, 3)),
                ("Relu3", ReLU())
            };
            conv = Sequential(modules);
            var convOutSize = GetConvOut(inputShape);
            var modules2 = new List<(string, Module<torch.Tensor, torch.Tensor>)>
            {
                ("Linear1", Linear(convOutSize, 512)),
                ("Relu4", ReLU()),
                ("Linear2", Linear(512, actions))
            };
            fc = Sequential(modules2);


            conv.to(new torch.Device(deviceType));
            fc.to(new torch.Device(deviceType));
            RegisterComponents();
        }


        public override torch.Tensor forward(torch.Tensor input)
        {
            var convOut = conv.forward(input).view(input.size(0), -1);
            var fcOut = fc.forward(convOut);
            return fcOut;
        }

        public int GetConvOut(long[] inputShape)
        {
            var arr = new List<long> {1};
            arr.AddRange(inputShape);
            var o = conv.forward(torch.zeros(arr.ToArray()));
            var shapes = o.size();
            var outSize = shapes.Aggregate((a, b) => a * b);
            return (int) outSize;
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                conv.Dispose();
                fc.Dispose();
                ClearModules();
            }

            base.Dispose(disposing);
        }
    }
}
