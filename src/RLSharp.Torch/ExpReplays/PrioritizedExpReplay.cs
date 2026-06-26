using RLSharp.Torch.Environs;

namespace RLSharp.Torch.ExpReplays
{
	/// <summary>
	///     Prioritized sampling from experience replay buffer.
	/// </summary>
	public class PrioritizedExpReplay : ReplayBuffer
	{
		/// <summary>
		/// </summary>
		/// <param name="c">Capacity of experience replay buffer (recommend 10^5 ~ 10^6).</param>
		public PrioritizedExpReplay(int capacity = 10000)
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
			var priorities = Buffers.Select(a => a.Priority).ToArray();
			if (priorities.Any(priority => !float.IsFinite(priority) || priority < 0))
				throw new InvalidOperationException("Replay priorities must be finite and non-negative.");
			if (priorities.All(priority => priority == 0))
				throw new InvalidOperationException("At least one replay priority must be greater than zero.");

			using var probs       = torch.from_array(priorities);
			using var sampled     = torch.multinomial(probs, batchsize, true);
			var       randomIndex = sampled.data<long>().ToArray();

			var steps = randomIndex
				.AsParallel()
				.Select(i => Buffers.ElementAt((int)i))
				.ToArray();

			return steps;
		}

		public void UpdatePriorities(IReadOnlyList<int> indices, IReadOnlyList<float> priorities)
		{
			ArgumentNullException.ThrowIfNull(indices);
			ArgumentNullException.ThrowIfNull(priorities);
			if (indices.Count != priorities.Count)
				throw new ArgumentException("Indices and priorities must have the same length.");

			var buffer = Buffers.ToArray();
			for (var i = 0; i < indices.Count; i++)
			{
				var index    = indices[i];
				var priority = priorities[i];
				if (index < 0 || index >= buffer.Length)
					throw new ArgumentOutOfRangeException(nameof(indices));
				if (!float.IsFinite(priority) || priority < 0)
					throw new ArgumentOutOfRangeException(nameof(priorities));
				buffer[index].Priority = priority;
			}
		}
	}
}