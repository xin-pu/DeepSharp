namespace DeepSharp.RL.Agents
{
    public sealed class PGN : Module<torch.Tensor, torch.Tensor>
    {
        private readonly Module<torch.Tensor, torch.Tensor> layers;

        public PGN(long obsSize, long hiddenSize, long actionNum, DeviceType deviceType = DeviceType.CUDA) :
            base("PolicyNet")
        {
            var modules = new List<(string, Module<torch.Tensor, torch.Tensor>)>
            {
                ("line1", Linear(obsSize, hiddenSize)),
                ("relu", ReLU()),
                ("line2", Linear(hiddenSize, actionNum)),
                ("softmax", Softmax(-1))
            };
            layers = Sequential(modules);
            layers.to(new torch.Device(deviceType));
            RegisterComponents();
        }

        public override torch.Tensor forward(torch.Tensor input)
        {
            return layers.forward(input.to_type(torch.ScalarType.Float32));
        }

        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                layers.Dispose();
                ClearModules();
            }

            base.Dispose(disposing);
        }
    }
}