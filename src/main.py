"""Main entry point for Chess Vision Engine."""

import click
from pathlib import Path

from src.detection import BoardDetector, PieceClassifier
from src.chess_logic import FENGenerator
from src.engine import StockfishWrapper
from src.utils.config import settings
from src.utils.logging_config import setup_logging, get_logger

logger = get_logger(__name__)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
def cli(verbose: bool) -> None:
    """Chess Vision Engine - Scan chessboards and analyze positions."""
    log_level = "DEBUG" if verbose else settings.log_level
    setup_logging(log_level)


@cli.command()
@click.option("--image", "-i", required=True, type=click.Path(exists=True), help="Path to chessboard image")
@click.option("--depth", "-d", default=20, help="Engine analysis depth")
@click.option("--output", "-o", type=click.Path(), help="Output file for analysis")
def analyze(image: str, depth: int, output: str | None) -> None:
    """Analyze a chessboard image."""
    logger.info(f"Analyzing image: {image}")

    try:
        detector = BoardDetector()
        classifier = PieceClassifier()
        fen_generator = FENGenerator()
        engine = StockfishWrapper()

        logger.info("Detecting board...")
        board_image = detector.detect_board(image)

        logger.info("Classifying pieces...")
        pieces = classifier.classify_pieces(board_image)

        logger.info("Generating FEN notation...")
        fen = fen_generator.generate(pieces)
        click.echo(f"Detected position: {fen}")

        logger.info(f"Running engine analysis (depth {depth})...")
        analysis = engine.analyze(fen, depth=depth)

        click.echo(f"\nAnalysis Results:")
        click.echo(f"  Best move: {analysis.best_move}")
        click.echo(f"  Evaluation: {analysis.score}")
        click.echo(f"  Principal variation: {' '.join(analysis.pv[:5])}")

        if output:
            _save_analysis(output, fen, analysis)
            click.echo(f"\nResults saved to: {output}")

    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option("--camera", "-c", default=0, help="Camera index")
@click.option("--depth", "-d", default=15, help="Engine analysis depth")
def live(camera: int, depth: int) -> None:
    """Start live webcam analysis."""
    logger.info(f"Starting live analysis with camera {camera}")

    try:
        import cv2
        from src.detection import BoardDetector, PieceClassifier
        from src.chess_logic import FENGenerator
        from src.engine import StockfishWrapper

        detector = BoardDetector()
        classifier = PieceClassifier()
        fen_generator = FENGenerator()
        engine = StockfishWrapper()

        cap = cv2.VideoCapture(camera)

        click.echo("Press 'q' to quit, 'a' to analyze current frame")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            cv2.imshow("Chess Vision Engine", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                try:
                    board = detector.detect_board_from_array(frame)
                    pieces = classifier.classify_pieces(board)
                    fen = fen_generator.generate(pieces)
                    analysis = engine.analyze(fen, depth=depth)

                    click.echo(f"\nFEN: {fen}")
                    click.echo(f"Best move: {analysis.best_move}")
                    click.echo(f"Evaluation: {analysis.score}")
                except Exception as e:
                    click.echo(f"Analysis failed: {e}")

        cap.release()
        cv2.destroyAllWindows()
        engine.close()

    except ImportError:
        raise click.ClickException("Live analysis requires OpenCV. Install with: pip install opencv-python")
    except Exception as e:
        logger.error(f"Live analysis failed: {e}")
        raise click.ClickException(str(e))


@cli.command()
@click.option("--port", "-p", default=8080, help="Server port")
@click.option("--host", "-h", default="0.0.0.0", help="Server host")
def serve(port: int, host: str) -> None:
    """Start the web server."""
    import uvicorn
    from src.ui.app import app

    logger.info(f"Starting server at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.argument("fen")
@click.option("--depth", "-d", default=20, help="Engine analysis depth")
def engine(fen: str, depth: int) -> None:
    """Analyze a FEN position with the engine."""
    try:
        engine = StockfishWrapper()
        result = engine.analyze(fen, depth=depth)

        click.echo(f"Position: {fen}")
        click.echo(f"Best move: {result.best_move}")
        click.echo(f"Evaluation: {result.score}")
        click.echo(f"Depth: {result.depth}")
        click.echo(f"PV: {' '.join(result.pv[:10])}")

        engine.close()

    except Exception as e:
        raise click.ClickException(str(e))


def _save_analysis(path: str, fen: str, analysis) -> None:
    """Save analysis results to file."""
    output_path = Path(path)

    with open(output_path, "w") as f:
        f.write(f"FEN: {fen}\n")
        f.write(f"Best Move: {analysis.best_move}\n")
        f.write(f"Evaluation: {analysis.score}\n")
        f.write(f"PV: {' '.join(analysis.pv)}\n")


if __name__ == "__main__":
    cli()