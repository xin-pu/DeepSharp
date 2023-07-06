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
            trainer.Train(0.85f, 1000, "", 20, 2);
        }

        [Fact]
        public void VeirfyReinforce()
        {
            var kArmedBandit = new KArmedBandit(new[] {0.4, 0.85, 0.75, 0.75}) {Gamma = 0.95f};
            var agent = new Reinforce(kArmedBandit);
            agent.Load("[Agent[Reinforce]]_230_0.90.st");
            var trainer = new RLTrainer(agent, Print);
            trainer.Val(1020);
        }

        [Fact]
        public void TestReinforce2()
        {
            var frozenlake = new Frozenlake(new[] {0.8f, 0.1f, 0.1f}) {Gamma = 0.95f};
            var agent = new Reinforce(frozenlake);
            var trainer = new RLTrainer(agent, Print);
            trainer.Train(0.9f, 1000, "", 20, 2);
        }


        [Fact]
        public void verifyReinforce2()
        {
            var frozenlake = new Frozenlake(new[] {0.8f, 0.1f, 0.1f}) {Gamma = 0.95f};
            var agent = new Reinforce(frozenlake);
            agent.Load("[Agent[Reinforce]]_304_0.90.st");
            var episode = agent.RunEpisode();
            Print(episode);
        }
    }
}
