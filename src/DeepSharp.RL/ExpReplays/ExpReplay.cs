using DeepSharp.RL.Environs;
using DeepSharp.RL.ExperienceSources;

namespace DeepSharp.RL.ExpReplays
{
	/// <summary>
	///     Experience replay for storing steps [State, Action, Reward, NextState].
	/// </summary>
	public abstract class ExpReplay
	{
		protected ExpReplay(int capacity = 10000)
		{
			Capacity = capacity;
			Buffers  = new Queue<Step>(capacity);
		}

		/// <summary>
		///     Capacity of experience replay buffer.
		/// </summary>
		public int Capacity { get; protected set; }

		/// <summary>
		///     Internal buffer queue.
		/// </summary>
		public Queue<Step> Buffers { get; set; }

		public int Size => Buffers.Count();

		/// <summary>
		///     Record one step [State, Action, Reward, NewState].
		/// </summary>
		/// <param name="step"></param>
		public virtual void Enqueue(Step step)
		{
			if (Buffers.Count == Capacity) Buffers.Dequeue();
			Buffers.Enqueue(step);
		}

		/// <summary>
		///     Record multiple steps.
		/// </summary>
		public void Enqueue(IEnumerable<Step> steps)
		{
			steps.ToList().ForEach(Enqueue);
		}

		protected abstract Step[] SampleSteps(int batchsize);

		public virtual void Enqueue(Episode episode)
		{
			Enqueue(episode.Steps);
		}

		public virtual ExperienceCase Sample(int batchsize)
		{
			var batchStep = SampleSteps(batchsize);

			// Get arrays from steps
			var stateArray     = batchStep.Select(a => a.PreState.Value!.unsqueeze(0)).ToArray();
			var actArray       = batchStep.Select(a => a.Action.Value!.unsqueeze(0)).ToArray();
			var rewardArray    = batchStep.Select(a => a.Reward.Value).ToArray();
			var stateNextArray = batchStep.Select(a => a.PostState.Value!.unsqueeze(0)).ToArray();
			var doneArray      = batchStep.Select(a => a.IsComplete).ToArray();

			// Convert to vstack tensors
			var state     = torch.vstack(stateArray);
			var actionV   = torch.vstack(actArray).to(torch.ScalarType.Int64);
			var reward    = torch.from_array(rewardArray).reshape(batchsize);
			var stateNext = torch.vstack(stateNextArray);
			var done      = torch.from_array(doneArray).reshape(batchsize);

			var excase = new ExperienceCase(state, actionV, reward, stateNext, done);
			return excase;
		}


		public void Clear()
		{
			Buffers.Clear();
		}
	}
}