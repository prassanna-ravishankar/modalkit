# Changelog

## Unreleased

### Breaking changes

- Requires `modal>=0.73.132`, the first release with `max_containers`, `scaledown_window` and `@modal.concurrent`. Modal 1.x removed the old parameter names, so earlier modalkit releases fail at decoration time on current modal.
- `deployment_config` keys renamed to match Modal's API. The old keys are rejected with an error naming the replacement:
  - `concurrency_limit` -> `max_containers`
  - `container_idle_timeout` -> `scaledown_window`
  - `allow_concurrent_inputs` -> `max_concurrent_inputs`
  - `allow_concurrent_inputs_handler` -> `max_concurrent_inputs_handler`
- Per-container concurrency is no longer passed to `@app.cls` / `@app.function`. Apply Modal's decorator with the new helpers:

  ```python
  @app.cls(**modal_config.get_app_cls_settings())
  @modal.concurrent(**modal_config.get_concurrent_settings())
  class MyApp(ModalService): ...

  @app.function(**modal_config.get_handler_settings())
  @modal.concurrent(**modal_config.get_handler_concurrent_settings())
  @modal.asgi_app(**modal_config.get_asgi_app_settings())
  def web_endpoints(): ...
  ```

  Without the decorator, Modal runs one input per container. The handler previously defaulted to 10.
- `ModalService` no longer defines `__init__`, so the `queue_backend` constructor argument is gone. Set it as a class attribute instead, the same way as `inference_implementation`:

  ```python
  class MyService(ModalService):
      inference_implementation = MyModel
      queue_backend = MyQueueBackend()
  ```

  Modal constructs service instances itself and rejects classes that combine a custom `__init__` with `modal.parameter()` fields, so the documented `@app.cls` pattern previously failed at decoration time.
