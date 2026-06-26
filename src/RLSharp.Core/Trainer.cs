namespace RLSharp.Core;

/// <summary>
///     Generic trainer base class.
///     Provides abstract Train/Val interface and event callback hooks.
///     Subclasses (e.g. RLSharp.Torch.Trainers.RLTrainer) implement concrete training logic.
/// </summary>
public abstract class Trainer
{
    /// <summary>
    ///     Training start callback.
    /// </summary>
    protected virtual void OnTrainStart()
    {
        TrainStarted?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>
    ///     Training end callback.
    /// </summary>
    protected virtual void OnTrainEnd()
    {
        TrainEnded?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>
    ///     Validation start callback.
    /// </summary>
    protected virtual void OnValStart(int epoch)
    {
        ValStarted?.Invoke(this, epoch);
    }

    /// <summary>
    ///     Validation end callback.
    /// </summary>
    protected virtual void OnValEnd(int epoch, float reward)
    {
        ValEnded?.Invoke(this, (epoch, reward));
    }

    /// <summary>
    ///     Save checkpoint callback.
    /// </summary>
    protected virtual void OnSaveStart()
    {
        SaveStarted?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>
    ///     Save complete callback.
    /// </summary>
    protected virtual void OnSaveEnd()
    {
        SaveEnded?.Invoke(this, EventArgs.Empty);
    }

    /// <summary>
    ///     Abstract train method.
    /// </summary>
    public abstract void Train();

    /// <summary>
    ///     Abstract validation method.
    /// </summary>
    public abstract void Val(int epoch);

    // --- Events ---

    public event EventHandler? TrainStarted;

    public event EventHandler? TrainEnded;

    public event EventHandler<int>? ValStarted;

    public event EventHandler<(int epoch, float reward)>? ValEnded;

    public event EventHandler? SaveStarted;

    public event EventHandler? SaveEnded;
}
