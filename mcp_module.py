import json
import os
import logging
from pathlib import Path
import shutil 
import unittest
from unittest.mock import patch, MagicMock, call 
import sys 

# OpenAI specific imports
import openai
from openai import (
    OpenAI, APIError, APITimeoutError, RateLimitError, APIConnectionError,
    AuthenticationError, APIStatusError, PermissionDeniedError, NotFoundError
)
import httpx 

# Configure logging with the new standardized format
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
)

# OpenAI API Key Setup
DUMMY_API_KEY = "sk-dummykey_for_testing_replace_if_real"
openai.api_key = os.getenv("OPENAI_API_KEY", DUMMY_API_KEY) 

effective_api_key = os.getenv("OPENAI_API_KEY")
if not effective_api_key:
    logging.warning(f"OPENAI_API_KEY environment variable not set. Using dummy API key: {DUMMY_API_KEY}.")
    effective_api_key = DUMMY_API_KEY

client = OpenAI(api_key=effective_api_key)

class MCP:
    CONTEXT_FILE = "mcp_context.json"
    def __init__(self):
        self.contexto = self._load_context() # self.contexto must always be a dict

    def _load_context(self) -> dict: # Ensure it always returns a dictionary
        try:
            if os.path.exists(self.CONTEXT_FILE):
                with open(self.CONTEXT_FILE, "r", encoding="utf-8") as f:
                    content = f.read()
                    if not content:
                        logging.info(f"Context file '{self.CONTEXT_FILE}' is empty. Initializing with empty context.")
                        return {}
                    
                    parsed_content = json.loads(content)
                    
                    if not isinstance(parsed_content, dict):
                        logging.warning(
                            f"Context file '{self.CONTEXT_FILE}' contained valid JSON, "
                            f"but the root element was of type '{type(parsed_content).__name__}', not a dictionary. "
                            "Initializing with empty context."
                        )
                        return {}
                    return parsed_content
            else:
                logging.info(f"Context file '{self.CONTEXT_FILE}' not found. Initializing with empty context.")
                return {}
        except json.JSONDecodeError:
            logging.warning(f"Error decoding JSON from '{self.CONTEXT_FILE}'. Initializing with empty context.")
            return {}
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading context from '{self.CONTEXT_FILE}': {e}. Initializing with empty context.")
            return {}

    def _save_context(self):
        try:
            with open(self.CONTEXT_FILE, "w", encoding="utf-8") as f:
                json.dump(self.contexto, f, indent=4, ensure_ascii=False)
        except Exception as e:
            logging.error(f"Error saving context to '{self.CONTEXT_FILE}': {e}")
    def registrar_evento(self, doc_name: str, evento_str: str):
        self.contexto[doc_name] = evento_str 
        self._save_context()
    def recuperar_contexto(self, doc_name: str) -> str | None:
        return self.contexto.get(doc_name)

def cargar_documentos(ruta: str) -> list[tuple[str, str]]:
    ruta_path = Path(ruta)
    if not ruta_path.exists():
        logging.error(f"La ruta especificada '{ruta}' no existe.")
        return []
    if not ruta_path.is_dir():
        logging.error(f"La ruta especificada '{ruta}' no es un directorio.")
        return []
    docs = []
    for archivo_path in ruta_path.glob("*.txt"): 
        try:
            with open(archivo_path, "r", encoding="utf-8") as f:
                contenido = f.read()
            docs.append((archivo_path.name, contenido))
            logging.info(f"Documento '{archivo_path.name}' cargado exitosamente.")
        except (IOError, UnicodeDecodeError) as e:
            logging.warning(f"No se pudo cargar el archivo '{archivo_path.name}': {e}")
    return docs

def analizar_riesgos(doc_content: str, perfil_cliente: dict, contexto_str: str | None, doc_name: str, model_name: str = "gpt-4") -> str:
    current_api_key = client.api_key
    if not current_api_key or current_api_key == DUMMY_API_KEY:
        logging.warning(f"OpenAI API key is '{current_api_key}'. Skipping actual API call for {doc_name}.")
        dummy_response_dict = {
            "resumen_general": f"Análisis simulado para {doc_name} (API key no configurada o es dummy).",
            "riesgos_detectados": [], "alertas_proactivas_sugeridas": []
        }
        return json.dumps(dummy_response_dict)
    contexto_dict = None
    if contexto_str:
        try:
            contexto_dict = json.loads(contexto_str)
        except json.JSONDecodeError:
            logging.warning(f"Failed to parse contexto_str for {doc_name}. Using None.")
    perfil_items = [f"- {key.replace('_', ' ').capitalize()}: {value}" for key, value in perfil_cliente.items()]
    perfil_str_para_prompt = "\n".join(perfil_items)
    prompt_content = f"""
Eres un asistente legal inteligente especializado en la detección de riesgos en documentos corporativos.
Analiza el siguiente documento legal:
--- DOCUMENTO ---
{doc_content}
--- FIN DOCUMENTO ---
Considera el siguiente perfil del cliente:
{perfil_str_para_prompt}
Y el siguiente contexto histórico de interacciones legales para este documento (si existe):
{json.dumps(contexto_dict) if contexto_dict else "No hay contexto previo disponible."}
Tu tarea es identificar los principales riesgos legales en el documento, clasificarlos y proponer recomendaciones.
Debes responder **estrictamente** en formato JSON. No incluyas ninguna explicación o texto fuera del JSON.
El JSON debe seguir esta estructura:
{{
  "resumen_general": "Una breve descripción general de los hallazgos.",
  "riesgos_detectados": [
    {{
      "descripcion_riesgo": "Descripción detallada del riesgo identificado.",
      "clausula_afectada": "Cláusula o sección específica del documento relacionada con el riesgo (si aplica).",
      "gravedad": "Clasificación de la gravedad del riesgo (ej: 'Alta', 'Media', 'Baja').",
      "recomendacion_legal": "Recomendación legal clara y accionable para mitigar el riesgo."
    }}
  ],
  "alertas_proactivas_sugeridas": [
    {{
      "tema_alerta": "Tema de la alerta (ej: 'Vencimiento de contrato', 'Incumplimiento normativo').",
      "condicion_disparo": "Condición que debería disparar la alerta.",
      "documento_relacionado": "{doc_name}"
    }}
  ]
}}
Si no se detectan riesgos, el array "riesgos_detectados" debe estar vacío.
Si el documento parece no ser un documento legal o está vacío, indica eso en el "resumen_general" y deja los arrays vacíos.
Asegúrate de que el JSON esté bien formado.
Analiza ahora y proporciona tu respuesta:
"""
    logging.info(f"Enviando solicitud a OpenAI API para: {doc_name} usando modelo {model_name}")
    error_dict_to_return = None
    try:
        response = client.chat.completions.create(
            model=model_name, messages=[{"role": "system", "content": "Eres un asistente legal experto."}, {"role": "user", "content": prompt_content}],
            temperature=0.5, response_format={"type": "json_object"}
        )
        analysis_content_str = response.choices[0].message.content
        if analysis_content_str is None:
             error_dict_to_return = {"error": True, "message": "Respuesta de API es None.", "type": "NoContentResponse"}
        else:
            try:
                json.loads(analysis_content_str) 
                logging.info(f"Respuesta JSON válida recibida de OpenAI API para {doc_name}.")
                return analysis_content_str
            except json.JSONDecodeError as e:
                error_dict_to_return = {"error": True, "message": f"Respuesta API no es JSON válido: {e}", "type": "APIResponseNotJSONError", "raw_response": analysis_content_str}
    except AuthenticationError as e: error_dict_to_return = {"error": True, "message": str(e), "type": "AuthenticationError"}
    except PermissionDeniedError as e: error_dict_to_return = {"error": True, "message": str(e), "type": "PermissionDeniedError"}
    except NotFoundError as e: error_dict_to_return = {"error": True, "message": str(e), "type": "NotFoundError"}
    except RateLimitError as e: error_dict_to_return = {"error": True, "message": str(e), "type": "RateLimitError"}
    except APITimeoutError as e: error_dict_to_return = {"error": True, "message": str(e), "type": "APITimeoutError"}
    except APIConnectionError as e: error_dict_to_return = {"error": True, "message": str(e), "type": "APIConnectionError"}
    except APIStatusError as e: error_dict_to_return = {"error": True, "message": str(e.message), "type": f"APIStatusError_{e.status_code}"}
    except APIError as e: error_dict_to_return = {"error": True, "message": str(e), "type": "APIError"}
    except Exception as e: error_dict_to_return = {"error": True, "message": str(e), "type": "UnexpectedError"}
    if error_dict_to_return:
        logging.error(f"Error en API para {doc_name} ({error_dict_to_return['type']}): {error_dict_to_return['message']}")
        return json.dumps(error_dict_to_return)
    return json.dumps({"error": True, "message": "Flujo inesperado en analizar_riesgos", "type": "InternalFlowError"})

mcp = MCP()

def ejecutar_analisis(ruta_docs: str, perfil_cliente: dict, model_name: str = "gpt-4") -> list[dict]:
    documentos = cargar_documentos(ruta_docs)
    resultados_procesados = []
    if not documentos:
        logging.warning("No se cargaron documentos.")
        return resultados_procesados
    for doc_name, doc_content in documentos:
        contexto_previo_str = mcp.recuperar_contexto(doc_name)
        informe_str = analizar_riesgos(doc_content, perfil_cliente, contexto_previo_str, doc_name, model_name=model_name)
        current_doc_result_entry = {}
        try:
            parsed_informe = json.loads(informe_str)
            current_doc_result_entry = {"documento": doc_name, "informe_json": parsed_informe, "parsing_error": False}
        except json.JSONDecodeError as e:
            logging.warning(f"No se pudo parsear el informe JSON de '{doc_name}': {e}. Contenido crudo: '{informe_str[:200]}...'")
            current_doc_result_entry = {"documento": doc_name, "informe_raw": informe_str, "parsing_error": True}
        resultados_procesados.append(current_doc_result_entry)
        mcp.registrar_evento(doc_name, informe_str)
        if not current_doc_result_entry.get("parsing_error") and current_doc_result_entry.get("informe_json"):
            informe_content = current_doc_result_entry["informe_json"]
            if "resumen_general" in informe_content and not informe_content.get("error"):
                sugerencias_alertas = informe_content.get("alertas_proactivas_sugeridas", [])
                if sugerencias_alertas:
                    logging.info(f"--- Alertas Proactivas Sugeridas para Documento: {doc_name} ---")
                    for i, alerta in enumerate(sugerencias_alertas):
                        tema = alerta.get('tema_alerta', 'No especificado')
                        condicion = alerta.get('condicion_disparo', 'No especificada')
                        doc_rel = alerta.get('documento_relacionado', doc_name) 
                        logging.info(f"  Alerta Sugerida {i+1}: Tema='{tema}', Condicion='{condicion}', Documento='{doc_rel}'")
                    logging.info(f"--- Fin de Alertas Sugeridas para {doc_name} ---")
    return resultados_procesados

# --- Test Harness ---
TEST_DOCS_DIR = Path("documentos_prueba_refactored") 
TEST_TEMP_DIR = Path("temp_test_files_mcp")     

def setup_test_environment(
    target_dir_path: Path, 
    num_docs: int = 0, 
    custom_content_list: list[tuple[str, str]] | None = None,
    cleanup_mcp_context: bool = True 
):
    try:
        if target_dir_path.exists(): shutil.rmtree(target_dir_path)
        target_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Error en setup_test_environment creando directorio '{target_dir_path}': {e}")
        raise
    if custom_content_list:
        for filename, content in custom_content_list:
            try: (target_dir_path / filename).write_text(content, encoding="utf-8")
            except IOError as e: logging.error(f"Error en setup_test_environment escribiendo archivo {filename} en '{target_dir_path}': {e}"); raise
    else:
        for i in range(1, num_docs + 1):
            try: (target_dir_path / f"documento_generico_{i}.txt").write_text(f"Contenido del documento genérico {i}.", encoding="utf-8")
            except IOError as e: logging.error(f"Error en setup_test_environment escribiendo archivo genérico {i} en '{target_dir_path}': {e}"); raise
    if cleanup_mcp_context:
        context_file = Path(MCP.CONTEXT_FILE)
        if context_file.exists(): context_file.unlink()
        global mcp
        mcp = MCP()

class TestMCP(unittest.TestCase):
    def setUp(self):
        setup_test_environment(TEST_DOCS_DIR, cleanup_mcp_context=True) 

    def test_context_file_is_valid_json_but_not_dictionary(self):
        context_filepath = Path(MCP.CONTEXT_FILE)
        invalid_json_roots = [
            ("JSON Array", "[]"),
            ("JSON String", "\"This is just a JSON string, not a dictionary.\""),
            ("JSON Number", "12345"),
            ("JSON Boolean", "true"),
            ("JSON Null", "null")
        ]
        for content_type, content_str in invalid_json_roots:
            with self.subTest(content_type=content_type):
                try:
                    with open(context_filepath, "w", encoding="utf-8") as f:
                        f.write(content_str)
                except IOError as e:
                    self.fail(f"Test setup failed: Could not write to context file {context_filepath}: {e}")
                expected_log_message_part = (
                    f"Context file '{MCP.CONTEXT_FILE}' contained valid JSON, "
                    f"but the root element was of type '{type(json.loads(content_str)).__name__}', not a dictionary."
                )
                with self.assertLogs(level='WARNING') as log_capture:
                    fresh_mcp_instance = MCP() 
                self.assertEqual(fresh_mcp_instance.contexto, {}, f"Contexto should be empty for {content_type}")
                self.assertTrue(
                    any(expected_log_message_part in log for log in log_capture.output),
                    f"Expected warning log not found for {content_type}. Captured: {log_capture.output}"
                )
                if context_filepath.exists():
                    context_filepath.unlink()

class TestCargarDocumentos(unittest.TestCase):
    def setUp(self):
        setup_test_environment(TEST_TEMP_DIR, cleanup_mcp_context=False)
    def tearDown(self):
        if TEST_TEMP_DIR.exists(): shutil.rmtree(TEST_TEMP_DIR)
    def test_non_existent_path(self):
        non_existent_path = str(TEST_TEMP_DIR / "ruta_inexistente_cargar")
        with self.assertLogs(level='ERROR') as log_capture:
            resultado = cargar_documentos(non_existent_path)
        self.assertEqual(resultado, [])
        self.assertTrue(any(f"La ruta especificada '{non_existent_path}' no existe." in msg for msg in log_capture.output))
    def test_path_is_file(self):
        file_path = TEST_TEMP_DIR / "un_archivo_test.txt"
        file_path.write_text("Soy un archivo, no un directorio.")
        with self.assertLogs(level='ERROR') as log_capture:
            resultado = cargar_documentos(str(file_path))
        self.assertEqual(resultado, [])
        self.assertTrue(any(f"La ruta especificada '{str(file_path)}' no es un directorio." in msg for msg in log_capture.output))
    def test_successful_loading_and_skipping(self):
        setup_test_environment(
            TEST_TEMP_DIR, custom_content_list=[
                ("doc1.txt", "Contenido de doc1"), ("doc2.txt", "Contenido de doc2"),
                ("unreadable.txt", ""), ("doc3.txt", "Contenido de doc3"),
                ("doc4.md", "Contenido de doc4.md (no .txt)")
            ], cleanup_mcp_context=False
        )
        unreadable_file_path = TEST_TEMP_DIR / "unreadable.txt"
        try:
            with open(unreadable_file_path, "wb") as f: f.write(b'\xff\xfe') 
        except IOError as e: self.skipTest(f"Could not create unreadable file for test setup: {e}")
        with self.assertLogs(level='WARNING') as log_capture:
            loaded_docs = cargar_documentos(str(TEST_TEMP_DIR))
        self.assertEqual(len(loaded_docs), 3) 
        doc_names = sorted([name for name, content in loaded_docs])
        self.assertEqual(doc_names, ["doc1.txt", "doc2.txt", "doc3.txt"])
        self.assertTrue(any(f"No se pudo cargar el archivo 'unreadable.txt'" in msg for msg in log_capture.output))

class TestAnalizarRiesgos(unittest.TestCase):
    def setUp(self):
        self.original_client_api_key = client.api_key
        client.api_key = "sk-testkey_analizar_riesgos"
        self.sample_profile_dict = {"sector": "Fintech", "region_operaciones": "LATAM", "antiguedad_empresa": "5 años"}
    def tearDown(self):
        client.api_key = self.original_client_api_key
    def _prepare_mock_error_response(self, status_code, error_message="Test Error", error_type="test_error", error_code=None):
        mock_headers = MagicMock(spec=dict); mock_headers.get.return_value = "test-request-id"
        mock_request = MagicMock(spec=httpx.Request); mock_request.url = "https://api.openai.com/v1/chat/completions"
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = status_code; mock_response.headers = mock_headers; mock_response.request = mock_request
        body = {"error": {"message": error_message, "type": error_type}}
        if error_code: body["error"]["code"] = error_code
        mock_response.json.return_value = body
        return mock_response, body
    @patch('mcp_module.client.chat.completions.create') # Corrected patch target
    def test_success_with_dict_profile(self, mock_api_call):
        expected_dict_output = {"resumen_general": "Éxito con perfil dict"}
        mock_api_call.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content=json.dumps(expected_dict_output)))])
        result_str = analizar_riesgos("Contenido", self.sample_profile_dict, None, "doc_ok.txt")
        self.assertIsInstance(result_str, str)
        self.assertEqual(json.loads(result_str), expected_dict_output)
        mock_api_call.assert_called_once()
        called_prompt = mock_api_call.call_args[1]['messages'][1]['content'] 
        expected_profile_str_in_prompt = ("- Sector: Fintech\n- Region operaciones: LATAM\n- Antiguedad empresa: 5 años")
        self.assertIn(expected_profile_str_in_prompt, called_prompt)
    @patch('mcp_module.client.chat.completions.create') # Corrected patch target
    def test_api_returns_malformed_json_string(self, mock_api_call):
        malformed_json_str = '{"summary": "OK", "details": "Missing quote}'
        mock_api_call.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content=malformed_json_str))])
        result_str = analizar_riesgos("Contenido", self.sample_profile_dict, None, "doc_malformed.txt")
        self.assertIsInstance(result_str, str)
        result_dict = json.loads(result_str)
        self.assertTrue(result_dict.get("error"))
        self.assertEqual(result_dict.get("type"), "APIResponseNotJSONError")
    @patch('mcp_module.client.chat.completions.create') # Corrected patch target
    def test_authentication_error_returns_json_string(self, mock_api_call):
        resp, body = self._prepare_mock_error_response(401, "Bad key", "auth_err")
        mock_api_call.side_effect = AuthenticationError(message="Bad key", response=resp, body=body)
        result_str = analizar_riesgos("Content", self.sample_profile_dict, None, "doc_auth.txt")
        self.assertIsInstance(result_str, str)
        result_dict = json.loads(result_str)
        self.assertTrue(result_dict.get("error"))
        self.assertEqual(result_dict.get("type"), "AuthenticationError")

class TestEjecutarAnalisis(unittest.TestCase):
    def setUp(self):
        self.original_client_api_key = client.api_key
        client.api_key = "sk-testkey_ejecutar_analisis"
        self.sample_profile_dict = {"sector_industrial": "Manufactura", "tamano_empresa": "Grande"}
        setup_test_environment(
            TEST_DOCS_DIR, custom_content_list=[
                ("contrato_test.txt", "Contrato de prueba."), ("poliza_test.txt", "Póliza de prueba."),
                ("factura_invalida.txt", "Factura con JSON inválido."), ("error_doc.txt", "Documento para simular error de análisis.")
            ], cleanup_mcp_context=True
        )
    def tearDown(self):
        client.api_key = self.original_client_api_key
        if TEST_DOCS_DIR.exists(): shutil.rmtree(TEST_DOCS_DIR)
    @patch('mcp_module.analizar_riesgos') # Corrected patch target
    def test_alert_logging_and_report_structure(self, mock_main_analizar_riesgos):
        alerts_list = [{"tema_alerta": "Vencimiento Próximo", "condicion_disparo": "Fecha < 30 días", "documento_relacionado": "contrato_test.txt"}]
        report_with_alerts_dict = {"resumen_general": "Contrato con alertas", "alertas_proactivas_sugeridas": alerts_list}
        report_without_alerts_dict = {"resumen_general": "Póliza sin alertas", "alertas_proactivas_sugeridas": []}
        report_as_error_dict = {"error": True, "message": "Fallo en análisis", "type": "SimulatedError"}
        report_invalid_json_str = "Esto no es JSON"
        def side_effect_for_analizar(doc_content, perfil_cliente, contexto_str, doc_name, model_name):
            if doc_name == "contrato_test.txt": return json.dumps(report_with_alerts_dict)
            elif doc_name == "poliza_test.txt": return json.dumps(report_without_alerts_dict)
            elif doc_name == "factura_invalida.txt": return report_invalid_json_str 
            elif doc_name == "error_doc.txt": return json.dumps(report_as_error_dict)
            return json.dumps({"error": True, "message": "Unknown test doc"})
        mock_main_analizar_riesgos.side_effect = side_effect_for_analizar
        with self.assertLogs(level='INFO') as log_capture:
            resultados = ejecutar_analisis(str(TEST_DOCS_DIR), self.sample_profile_dict)
        self.assertEqual(mock_main_analizar_riesgos.call_count, 4)
        self.assertEqual(len(resultados), 4)
        res_contrato = next(r for r in resultados if r["documento"] == "contrato_test.txt")
        self.assertFalse(res_contrato["parsing_error"])
        self.assertEqual(res_contrato["informe_json"], report_with_alerts_dict)
        self.assertTrue(any("Alertas Proactivas Sugeridas para Documento: contrato_test.txt" in msg for msg in log_capture.output))
    @patch('mcp_module.analizar_riesgos') # Corrected patch target
    def test_persistence_with_dict_profile(self, mock_main_analizar_riesgos):
        test_profile = {"tipo_cliente": "Gubernamental"}
        def side_effect_run1(doc_content, perfil_cliente, contexto_str, doc_name, model_name):
            self.assertEqual(perfil_cliente, test_profile) 
            self.assertIsNone(contexto_str)
            if doc_name == "contrato_test.txt": return json.dumps({"resumen_general": f"Resultado {doc_name} v1", "alertas_proactivas_sugeridas": [{"tema_alerta": "Contexto Run1"}]})
            return json.dumps({"resumen_general": f"Resultado {doc_name} v1"})
        mock_main_analizar_riesgos.side_effect = side_effect_run1
        ejecutar_analisis(str(TEST_DOCS_DIR), test_profile, "model-v1-dict")
        global mcp; mcp = MCP() 
        mock_main_analizar_riesgos.reset_mock()
        def side_effect_run2(doc_content, perfil_cliente, contexto_str, doc_name, model_name):
            self.assertEqual(perfil_cliente, test_profile) 
            expected_prev_context_dict_content = {"resumen_general": f"Resultado {doc_name} v1"}
            if doc_name == "contrato_test.txt": expected_prev_context_dict_content["alertas_proactivas_sugeridas"] = [{"tema_alerta": "Contexto Run1"}]
            self.assertEqual(json.loads(contexto_str), expected_prev_context_dict_content)
            return json.dumps({"resumen_general": f"Resultado {doc_name} v2"})
        mock_main_analizar_riesgos.side_effect = side_effect_run2
        ejecutar_analisis(str(TEST_DOCS_DIR), test_profile, "model-v2-dict")
        self.assertEqual(json.loads(mcp.recuperar_contexto("contrato_test.txt"))["resumen_general"], "Resultado contrato_test.txt v2")

if __name__ == '__main__':
    # --- Example Pipeline Execution ---
    print("\n" + "="*50)
    print("INICIANDO EJECUCIÓN DE EJEMPLO DEL SERVICIO PREDICTIVO")
    print("="*50 + "\n")
    DOCS_DIR_EJEMPLO = Path("documentos_ejemplo_analisis")
    example_mcp_context_file = Path(MCP.CONTEXT_FILE)
    if example_mcp_context_file.exists():
        try: example_mcp_context_file.unlink(); logging.info(f"Archivo de contexto MCP '{example_mcp_context_file}' eliminado.")
        except OSError as e: logging.warning(f"No se pudo eliminar '{example_mcp_context_file}': {e}")
    mcp = MCP() 
    try:
        logging.info(f"Configurando entorno de demostración en '{DOCS_DIR_EJEMPLO}'...")
        setup_test_environment( DOCS_DIR_EJEMPLO, custom_content_list=[
                ("contrato_demostracion.txt", "Contrato demostración."), ("email_importante.txt", "Email urgente."),
                ("minuta_reunion.txt", "Minuta reunión."), ("documento_sin_alertas.txt", "Doc sin alertas."),
                ("documento_con_error_analisis.txt", "Doc error análisis."), ("documento_respuesta_no_json.txt", "Doc no JSON.")
            ], cleanup_mcp_context=False
        )
        logging.info(f"Entorno de demostración configurado en '{DOCS_DIR_EJEMPLO}'.")
    except Exception as e: logging.error(f"CRITICAL: Setup ejemplo falló: {e}", exc_info=True); sys.exit(1)
    perfil_cliente_ejemplo = { "nombre_empresa": "Innovaciones Globales S.A.", "sector_industrial": "Consultoría Tecnológica"}
    logging.info(f"Usando perfil de cliente: {json.dumps(perfil_cliente_ejemplo, indent=2, ensure_ascii=False)}")
    modelo_llm_ejemplo = "gpt-3.5-turbo" 
    logging.info(f"Usando modelo LLM: {modelo_llm_ejemplo}")
    if client.api_key == DUMMY_API_KEY: logging.warning("ADVERTENCIA: API Key no configurada o dummy. Usando respuestas simuladas.")
    resultados_analisis = []
    try:
        logging.info(f"Iniciando análisis de docs en '{DOCS_DIR_EJEMPLO}'...")
        resultados_analisis = ejecutar_analisis(str(DOCS_DIR_EJEMPLO), perfil_cliente_ejemplo, model_name=modelo_llm_ejemplo)
        logging.info("Análisis de documentos completado.")
    except Exception as e: logging.error(f"CRITICAL: Análisis de ejemplo falló: {e}", exc_info=True)
    print("\n" + "="*50 + "\nRESULTADOS DEL ANÁLISIS DE RIESGOS (EJEMPLO)\n" + "="*50)
    if resultados_analisis:
        for res_doc in resultados_analisis:
            print(f"\n--- Documento: {res_doc['documento']} ---")
            if res_doc.get('parsing_error'):
                print(f"  ERROR: Informe no pudo ser parseado como JSON. Contenido Raw: {res_doc.get('informe_raw', 'N/A')}")
            elif res_doc.get('informe_json'):
                informe = res_doc['informe_json']
                if informe.get('error'): print("  ERROR: Análisis retornó un error:")
                else: print("  Informe del Análisis:")
                print(json.dumps(informe, indent=2, ensure_ascii=False))
            else: print("  ERROR: No se encontró contenido de informe.")
            print("--- Fin del Documento ---")
    else: print("No se obtuvieron resultados del análisis.")
    if DOCS_DIR_EJEMPLO.exists():
        try: shutil.rmtree(DOCS_DIR_EJEMPLO); logging.info(f"Directorio ejemplo '{DOCS_DIR_EJEMPLO}' eliminado.")
        except OSError as e: logging.error(f"Error eliminando directorio ejemplo '{DOCS_DIR_EJEMPLO}': {e}")
    print("\nINFO: Fin de la ejecución de ejemplo.\nPara ejecutar tests: python -m unittest mcp_module.py")
