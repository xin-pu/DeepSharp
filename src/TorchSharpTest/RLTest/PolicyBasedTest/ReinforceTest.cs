using DeepSharp.RL.Agents;
using DeepSharp.RL.Environs;
using DeepSharp.RL.Trainers;

namespace TorchSharpTest.RLTest.PolicyBasedTest
{
    public class ReinforceTest:AbstractTest
    {
        public ReinforceTest(ITestOutputHelper testOutputHelper) 
            : base(testOutputHelper)
        {
        }

        [Fact]
        public void TestReinforce()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.75, 0.75}) {Gamma = 0.95f};
            var agent = new Reinforce(kArmedBandit);
            var trainer = new RLTrainer(agent, Print);
            trainer.Train(0.9f, 500, "", 20, 2);
        }

        [Fact]
        public void VeirfyReinforce()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.75, 0.75}) {Gamma = 0.95f};
            var agent = new Reinforce(kArmedBandit);
            agent.Load("[Agent[Reinforce]]_230_0.90.st");
            var episode = agent.RunEpisode();
            Print(episode);
        }
    }
}
