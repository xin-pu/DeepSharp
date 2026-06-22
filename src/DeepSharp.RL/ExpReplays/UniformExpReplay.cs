using DeepSharp.RL.Environs;

namespace DeepSharp.RL.ExpReplays
{
	/// <summary>
	///     Uniform random sampling from experience replay buffer.
	/// </summary>
	public class UniformExpReplay : ExpReplay
	{
		/// <summary>
		/// </summary>
		/// <param name="c">Capacity of experience replay buffer (recommend 10^5 ~ 10^6).</param>
		public UniformExpReplay(int capacity = 10000)
			: base(capacity)
		{
		}


		/// <summary>
		///     Uniform sample batch size steps from Queue
		/// </summary>
		/// <param name="batchsize">batch size</param>
		/// <returns></returns>
		protected override Step[] SampleSteps(int batchsize)
		{
			var randomIndex = torch.randint(0, Size, new[] { batchsize }).data<long>().ToArray();

			var steps = randomIndex
				.Select(i => Buffers.ElementAt((int)i))
				.ToArray();

			return steps;
		}
	}
}