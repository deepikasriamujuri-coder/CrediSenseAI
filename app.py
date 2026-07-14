import os
import logging
from logging.handlers import RotatingFileHandler
from flask import Flask, request, jsonify, render_template

from config import config_by_name
from src.prediction import CreditPredictor
from src.explainability import CreditRiskExplainer

# Global instances loaded at startup
predictor_instance = None
explainer_instance = None
PREDICTION_AVAILABLE = False
SHAP_AVAILABLE = False
SHAP_ERROR_MSG = ""

def create_app(config_name=None):
    app = Flask(__name__)
    
    # Select configuration
    if not config_name:
        config_name = os.environ.get("FLASK_ENV", "development")
    app.config.from_object(config_by_name.get(config_name, config_by_name["development"]))
    
    # Configure Logging
    os.makedirs("logs", exist_ok=True)
    log_format = logging.Formatter('%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]')
    
    # File Handler
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(log_format)
    file_handler.setLevel(app.config["LOG_LEVEL"])
    app.logger.addHandler(file_handler)
    
    # Console Handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_format)
    stream_handler.setLevel(app.config["LOG_LEVEL"])
    app.logger.addHandler(stream_handler)
    
    app.logger.setLevel(app.config["LOG_LEVEL"])
    app.logger.info("CrediSense AI startup sequence initiated.")
    
    global predictor_instance, explainer_instance, PREDICTION_AVAILABLE, SHAP_AVAILABLE, SHAP_ERROR_MSG
    
    # 1. Startup model and metadata loading strategy
    model_path = app.config["MODEL_PATH"]
    metadata_path = app.config["METADATA_PATH"]
    
    # Verify Prediction Model presence and deserialization compatibility
    try:
        if not os.path.exists(model_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError("Model binary or metadata JSON file missing.")
            
        predictor_instance = CreditPredictor(model_path=model_path, metadata_path=metadata_path)
        PREDICTION_AVAILABLE = True
        app.logger.info("Production Predictor initialized successfully.")
    except Exception as e:
        PREDICTION_AVAILABLE = False
        app.logger.error(f"CRITICAL: Prediction model initialization failed: {str(e)}")
        
    # Verify SHAP explainer compatibility
    if PREDICTION_AVAILABLE:
        try:
            # Startup compatibility verification test (monkeypatched TreeExplainer initialization)
            explainer_instance = CreditRiskExplainer(model_path=model_path, metadata_path=metadata_path)
            SHAP_AVAILABLE = True
            app.logger.info("SHAP CreditRiskExplainer initialized successfully.")
        except Exception as e:
            SHAP_AVAILABLE = False
            SHAP_ERROR_MSG = str(e)
            app.logger.warning(
                f"SHAP explanation service degraded or unavailable. Explainer failed to initialize: {SHAP_ERROR_MSG}"
            )
            
    # Centralized Error Handlers
    @app.errorhandler(400)
    def bad_request(error):
        msg = error.description if hasattr(error, "description") else "Bad request"
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": {"type": "validation_error", "message": msg}}), 400
        return render_template("error.html", title="Bad Request (400)", message=msg), 400
        
    @app.errorhandler(404)
    def not_found(error):
        msg = "The requested resource could not be found."
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": {"type": "not_found", "message": msg}}), 404
        return render_template("error.html", title="Page Not Found (404)", message=msg), 404
        
    @app.errorhandler(405)
    def method_not_allowed(error):
        msg = "The HTTP method is not allowed for this endpoint."
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": {"type": "method_not_allowed", "message": msg}}), 405
        return render_template("error.html", title="Method Not Allowed (405)", message=msg), 405
        
    @app.errorhandler(413)
    def request_entity_too_large(error):
        msg = "The uploaded payload size exceeds the server's limit."
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": {"type": "payload_too_large", "message": msg}}), 413
        return render_template("error.html", title="Payload Too Large (413)", message=msg), 413
        
    @app.errorhandler(500)
    def internal_server_error(error):
        # Do not leak stack traces or local paths to users
        msg = "An unexpected server error occurred."
        app.logger.error(f"Internal Server Error: {str(error)}")
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": {"type": "internal_error", "message": msg}}), 500
        return render_template("error.html", title="Internal Server Error (500)", message=msg), 500
        
    @app.errorhandler(503)
    def service_unavailable(error):
        msg = "The prediction service is currently unavailable."
        if request.path.startswith("/api/"):
            return jsonify({"success": False, "error": {"type": "service_unavailable", "message": msg}}), 503
        return render_template("error.html", title="Service Unavailable (503)", message=msg), 503

    # Helper function to check predictor availability
    def check_prediction_service():
        if not PREDICTION_AVAILABLE:
            app.logger.warning("Attempted request while prediction model is unavailable.")
            from werkzeug.exceptions import ServiceUnavailable
            raise ServiceUnavailable("Prediction model is unavailable.")
            
    # Browser routes
    @app.route("/", methods=["GET"])
    def index():
        return render_template("index.html")
        
    @app.route("/predict", methods=["POST"])
    def predict_form():
        check_prediction_service()
        
        # Extract form parameters
        form_data = {}
        for key in request.form:
            form_data[key] = request.form[key]
            
        # Parse fields to correct datatypes where applicable (e.g. empty or missing)
        # Validation and inference
        is_valid, err_msg, cleaned_data = predictor_instance.validate_input(form_data)
        if not is_valid:
            return render_template("error.html", title="Validation Error", message=err_msg), 400
            
        pred_res = predictor_instance.predict(cleaned_data)
        if not pred_res["success"]:
            return render_template("error.html", title="Prediction Error", message=pred_res["error"]), 500
            
        # explanation handling with graceful degradation
        explanation = None
        explanation_available = False
        
        if SHAP_AVAILABLE:
            try:
                explanation = explainer_instance.explain(cleaned_data, precomputed_prediction=pred_res)
                explanation_available = explanation["success"]
            except Exception as e:
                app.logger.warning(f"Failed to generate SHAP explanation for form request: {str(e)}")
                explanation_available = False
                
        return render_template(
            "result.html", 
            prediction=pred_res, 
            explanation=explanation, 
            explanation_available=explanation_available,
            shap_error=SHAP_ERROR_MSG
        )
        
    @app.route("/about", methods=["GET"])
    def about():
        return render_template("about.html")
        
    # REST API Routes
    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        check_prediction_service()
        
        # Enforce strict content-type handling for APIs
        if not request.is_json:
            return jsonify({"success": False, "error": {"type": "validation_error", "message": "Content-Type must be application/json"}}), 400
            
        raw_json = request.get_json()
        is_valid, err_msg, cleaned_data = predictor_instance.validate_input(raw_json)
        if not is_valid:
            return jsonify({"success": False, "error": {"type": "validation_error", "message": err_msg}}), 400
            
        pred_res = predictor_instance.predict(cleaned_data)
        if not pred_res["success"]:
            return jsonify({"success": False, "error": {"type": "internal_error", "message": pred_res["error"]}}), 500
            
        # Clean keys from predictor response to match API Schema spec
        return jsonify({
            "success": True,
            "prediction": {
                "risk_probability": pred_res["risk_probability"],
                "risk_percentage": pred_res["risk_percentage"],
                "predicted_class": pred_res["predicted_class"],
                "predicted_label": pred_res["predicted_label"],
                "risk_category": pred_res["risk_category"],
                "classification_threshold": pred_res["classification_threshold"],
                "risk_boundaries": predictor_instance.metadata["risk_category_boundaries"],
                "model_name": pred_res["model_name"],
                "disclaimer": pred_res["disclaimer"]
            }
        }), 200
        
    @app.route("/api/explain", methods=["POST"])
    def api_explain():
        check_prediction_service()
        
        if not request.is_json:
            return jsonify({"success": False, "error": {"type": "validation_error", "message": "Content-Type must be application/json"}}), 400
            
        raw_json = request.get_json()
        is_valid, err_msg, cleaned_data = predictor_instance.validate_input(raw_json)
        if not is_valid:
            return jsonify({"success": False, "error": {"type": "validation_error", "message": err_msg}}), 400
            
        # Degrade gracefully if SHAP is unavailable
        if not SHAP_AVAILABLE:
            pred_res = predictor_instance.predict(cleaned_data)
            return jsonify({
                "success": True,
                "explanation_available": False,
                "message": f"SHAP explanations are currently unavailable: {SHAP_ERROR_MSG}",
                "prediction_summary": {
                    "risk_probability": pred_res["risk_probability"],
                    "risk_category": pred_res["risk_category"]
                }
            }), 200
            
        try:
            explanation_res = explainer_instance.explain(cleaned_data)
            if not explanation_res["success"]:
                return jsonify({"success": False, "error": {"type": "internal_error", "message": explanation_res["error"]}}), 500
                
            return jsonify({
                "success": True,
                "explanation_available": True,
                "explanation": {
                    "explanation_scope": explanation_res["explanation_scope"],
                    "base_value": explanation_res["base_value"],
                    "top_risk_increasing_factors": explanation_res["top_risk_increasing_factors"],
                    "top_risk_decreasing_factors": explanation_res["top_risk_decreasing_factors"],
                    "disclaimer": explanation_res["disclaimer"]
                }
            }), 200
        except Exception as e:
            app.logger.warning(f"Unexpected explanation generation crash: {str(e)}")
            return jsonify({
                "success": True,
                "explanation_available": False,
                "message": f"SHAP explanations are degraded: {str(e)}"
            }), 200
            
    @app.route("/api/predict-with-explanation", methods=["POST"])
    def api_predict_with_explanation():
        check_prediction_service()
        
        if not request.is_json:
            return jsonify({"success": False, "error": {"type": "validation_error", "message": "Content-Type must be application/json"}}), 400
            
        raw_json = request.get_json()
        is_valid, err_msg, cleaned_data = predictor_instance.validate_input(raw_json)
        if not is_valid:
            return jsonify({"success": False, "error": {"type": "validation_error", "message": err_msg}}), 400
            
        pred_res = predictor_instance.predict(cleaned_data)
        if not pred_res["success"]:
            return jsonify({"success": False, "error": {"type": "internal_error", "message": pred_res["error"]}}), 500
            
        # Explanations with graceful degradation
        explanation_available = False
        explanation_body = None
        message = ""
        
        if SHAP_AVAILABLE:
            try:
                exp_res = explainer_instance.explain(cleaned_data, precomputed_prediction=pred_res)
                if exp_res["success"]:
                    explanation_available = True
                    explanation_body = {
                        "explanation_scope": exp_res["explanation_scope"],
                        "base_value": exp_res["base_value"],
                        "top_risk_increasing_factors": exp_res["top_risk_increasing_factors"],
                        "top_risk_decreasing_factors": exp_res["top_risk_decreasing_factors"],
                        "disclaimer": exp_res["disclaimer"]
                    }
                else:
                    message = exp_res["error"]
            except Exception as e:
                explanation_available = False
                message = str(e)
        else:
            message = f"SHAP service unavailable: {SHAP_ERROR_MSG}"
            
        return jsonify({
            "success": True,
            "prediction": {
                "risk_probability": pred_res["risk_probability"],
                "risk_percentage": pred_res["risk_percentage"],
                "predicted_class": pred_res["predicted_class"],
                "predicted_label": pred_res["predicted_label"],
                "risk_category": pred_res["risk_category"],
                "classification_threshold": pred_res["classification_threshold"],
                "risk_boundaries": predictor_instance.metadata["risk_category_boundaries"],
                "model_name": pred_res["model_name"],
                "disclaimer": pred_res["disclaimer"]
            },
            "explanation_available": explanation_available,
            "explanation": explanation_body,
            "explanation_message": message if not explanation_available else None
        }), 200
        
    @app.route("/api/model-info", methods=["GET"])
    def api_model_info():
        check_prediction_service()
        metadata = predictor_instance.metadata
        
        # Do not expose absolute local filesystem paths, env vars, or secrets
        return jsonify({
            "success": True,
            "model_info": {
                "project_name": metadata["project_name"],
                "model_type": metadata["model_type"],
                "calibration_method": metadata["calibration_method"],
                "number_of_features": len(metadata["feature_names"]),
                "feature_names": metadata["feature_names"],
                "classification_threshold": metadata["classification_threshold"],
                "risk_category_boundaries": metadata["risk_category_boundaries"],
                "test_metrics": metadata["final_held_out_test_metrics"],
                "SHAP_explanation_scope": "underlying_xgboost_model_ensemble",
                "library_versions": metadata["model_library_versions"]
            }
        }), 200
        
    @app.route("/health", methods=["GET"])
    def health():
        status_code = 200
        status_str = "healthy"
        
        if not PREDICTION_AVAILABLE:
            status_code = 503
            status_str = "unhealthy"
            
        return jsonify({
            "status": status_str,
            "prediction_service_available": PREDICTION_AVAILABLE,
            "explanation_service_available": SHAP_AVAILABLE,
            "metadata_available": PREDICTION_AVAILABLE,
            "application_version": "1.0.0"
        }), status_code
        
    return app

if __name__ == "__main__":
    app_instance = create_app()
    app_instance.run(host="0.0.0.0", port=5000)
