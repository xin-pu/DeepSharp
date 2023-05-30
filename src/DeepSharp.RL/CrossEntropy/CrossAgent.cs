using DeepSharp.RL.Models;
using DeepSharp.RL.Policies;

namespace DeepSharp.RL.CrossEntropy
{
    public class CrossAgent : Agent
    {
        public override IPolicy LearnPolicy(Reward[] rewards)
        {
            throw new NotImplementedException();
        }

        public override torch.Tensor RunPolicy(State state)
        {
            throw new NotImplementedException();
        }
    }


    /// <summary>
    ///     This is demo net to guide how to create a new Module
    /// </summary>
    public sealed class Net : Module<torch.Tensor, torch.Tensor>
    {
        private readonly Module<torch.Tensor, torch.Tensor> layers;

        public Net(int obsSize, int hiddenSize, int actionNum) : base("Net")
        {
            var modules = new List<(string, Module<torch.Tensor, torch.Tensor>)>
            {
                ("line1", Linear(obsSize, hiddenSize)),
                ("relu", ReLU()),
                ("line2", Linear(hiddenSize, actionNum))
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
