# Changelog

## Unreleased

### Breaking changes

- `ModalService` no longer defines `__init__`, so the `queue_backend` constructor argument is gone. Set it as a class attribute instead, the same way as `inference_implementation`:

  ```python
  class MyService(ModalService):
      inference_implementation = MyModel
      queue_backend = MyQueueBackend()
  ```

  Modal constructs service instances itself and rejects classes that combine a custom `__init__` with `modal.parameter()` fields, so the documented `@app.cls` pattern previously failed at decoration time.
