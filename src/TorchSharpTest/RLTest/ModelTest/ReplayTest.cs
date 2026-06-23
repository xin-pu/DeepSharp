using DeepSharp.RL.Environs;
using DeepSharp.RL.ExpReplays;

namespace TorchSharpTest.RLTest.ModelTest
{
	public class ReplayTest
	{
		[Fact]
		public void ReplayRejectsInvalidCapacityAndEmptySampling()
		{
			var create = () => new UniformExpReplay(0);
			create.Should().Throw<ArgumentOutOfRangeException>();

			var replay = new UniformExpReplay(2);
			var sample = () => replay.Sample(1);
			sample.Should().Throw<InvalidOperationException>();
		}

		[Fact]
		public void PrioritizedReplayValidatesAndUpdatesPriorities()
		{
			var replay = new PrioritizedExpReplay(2);
			replay.Enqueue(CreateStep(1));
			replay.Enqueue(CreateStep(1));

			replay.UpdatePriorities([0, 1], [0.25f, 0.75f]);
			replay.Buffers.Select(step => step.Priority).Should().Equal(0.25f, 0.75f);

			var update = () => replay.UpdatePriorities([2], [1f]);
			update.Should().Throw<ArgumentOutOfRangeException>();
		}

		private static Step CreateStep(float priority)
		{
			return new Step(
				new Observation(tensor(new[] { 1f })),
				new Act(tensor(0)),
				new Observation(tensor(new[] { 0f })),
				new Reward(1),
				priority: priority);
		}
	}
}