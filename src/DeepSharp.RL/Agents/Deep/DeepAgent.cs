using DeepSharp.RL.Environs;

namespace DeepSharp.RL.Agents.Deep
{
	/// <summary>
	///     Neural network agent base class.
	///     Provides optimizer, loss function, main network and default Save/Load.
	/// </summary>
	public abstract class DeepAgent : Agent, INetworkAgent
	{
		protected DeepAgent(Environ<Space, Space> env, string name)
			: base(env, name)
		{
		}

		/// <summary>
		///     Loss function.
		/// </summary>
		public Loss<torch.Tensor, torch.Tensor, torch.Tensor> Loss { get; protected set; } = null!;

		/// <summary>
		///     Main network (overridden by subclasses to expose the corresponding network module).
		/// </summary>
		public abstract Module<torch.Tensor, torch.Tensor> MainNet { get; }

		/// <summary>
		///     Optimizer.
		/// </summary>
		public Optimizer Optimizer { get; protected set; } = null!;

		/// <summary>
		///     Save checkpoint (model + optimizer state).
		/// </summary>
		public virtual void SaveCheckpoint(string dir)
		{
			Directory.CreateDirectory(dir);
			var modelPath = Path.Combine(dir, "model.dat");
			Save(modelPath);
		}

		/// <summary>
		///     Load checkpoint (model + optimizer state).
		/// </summary>
		public virtual void LoadCheckpoint(string dir)
		{
			var modelPath = Path.Combine(dir, "model.dat");
			Load(modelPath);
		}

		/// <summary>
		///     Default save: saves the main network.
		/// </summary>
		public override void Save(string path)
		{
			if (File.Exists(path)) File.Delete(path);
			MainNet.save(path);
		}

		/// <summary>
		///     Default load: loads the main network.
		/// </summary>
		public override void Load(string path)
		{
			MainNet.load(path);
		}
	}
}