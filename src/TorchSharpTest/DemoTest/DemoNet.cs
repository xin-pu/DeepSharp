namespace TorchSharpTest.DemoTest
{
    /// <summary>
    ///     This is demo net to guide how to create a new Module
    /// </summary>
    public sealed class DemoNet : Module<torch.Tensor, torch.Tensor>
    {
        private readonly Module<torch.Tensor, torch.Tensor> layers;

        public DemoNet(int obsSize, int actionNum) : base("Net")
        {
            var modules = new List<(string, Module<torch.Tensor, torch.Tensor>)>
            {
                ("line1", Linear(obsSize, 10)),
                ("line2", Linear(10, actionNum))
            };
            layers = Sequential(modules);
            RegisterComponents();
        }


        public override torch.Tensor forward(torch.Tensor input)
        {
            return layers.forward(input);
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