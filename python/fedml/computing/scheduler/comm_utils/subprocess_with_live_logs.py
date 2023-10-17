import os
import selectors
import subprocess
import threading


class Popen(subprocess.Popen):
    LATEST_ERROR_LINE_NUM = 100

    def __init__(self, args, bufsize=-1, executable=None,
                 stdin=None, stdout=None, stderr=None,
                 preexec_fn=None, close_fds=True,
                 shell=False, cwd=None, env=None, universal_newlines=None,
                 startupinfo=None, creationflags=0,
                 restore_signals=True, start_new_session=False,
                 pass_fds=(), *, encoding=None, errors=None, text=None):
        super().__init__(args, bufsize=bufsize, executable=executable,
                         stdin=stdin, stdout=stdout, stderr=stderr,
                         preexec_fn=preexec_fn, close_fds=close_fds,
                         shell=shell, cwd=cwd, env=env, universal_newlines=universal_newlines,
                         startupinfo=startupinfo, creationflags=creationflags,
                         restore_signals=restore_signals, start_new_session=start_new_session,
                         pass_fds=pass_fds, encoding=encoding, errors=errors, text=text)

    def communicate(self, input=None, timeout=None,
                    data_arrived_callback=None, error_processor=None, should_write_log=True):
        """Interact with process: Send data to stdin and close it.
        Read data from stdout and stderr, until end-of-file is
        reached.  Wait for process to terminate.

        The optional "input" argument should be data to be sent to the
        child process, or None, if no data should be sent to the child.
        communicate() returns a tuple (stdout, stderr).

        By default, all communication is in bytes, and therefore any
        "input" should be bytes, and the (stdout, stderr) will be bytes.
        If in text mode (indicated by self.text_mode), any "input" should
        be a string, and (stdout, stderr) will be strings decoded
        according to locale encoding, or by "encoding" if set. Text mode
        is triggered by setting any of text, encoding, errors or
        universal_newlines.
        """

        if self._communication_started and input:
            raise ValueError("Cannot send input after starting communication")

        # Optimization: If we are not worried about timeouts, we haven't
        # started communicating, and we have one or zero pipes, using select()
        # or threads is unnecessary.
        if (timeout is None and not self._communication_started and
                [self.stdin, self.stdout, self.stderr].count(None) >= 2):
            stdout = None
            stderr = None
            if self.stdin:
                self._stdin_write(input)
            elif self.stdout:
                stdout = self.stdout.read()
                self.stdout.close()
            elif self.stderr:
                stderr = self.stderr.read()
                self.stderr.close()
            self.wait()
        else:
            if timeout is not None:
                endtime = subprocess._time() + timeout
            else:
                endtime = None

            try:
                str_stdout, str_stderr, stdout, stderr, latest_lines_err_list = self._communicate(
                    input, endtime, timeout, data_arrived_callback, error_processor=error_processor,
                    should_write_log=should_write_log)
            except KeyboardInterrupt:
                # https://bugs.python.org/issue25942
                # See the detailed comment in .wait().
                if timeout is not None:
                    sigint_timeout = min(self._sigint_wait_secs,
                                         self._remaining_time(endtime))
                else:
                    sigint_timeout = self._sigint_wait_secs
                self._sigint_wait_secs = 0  # nothing else should wait.
                try:
                    self._wait(timeout=sigint_timeout)
                except subprocess.TimeoutExpired:
                    pass
                raise  # resume the KeyboardInterrupt

            finally:
                self._communication_started = True

            try:
                sts = self.wait(timeout=self._remaining_time(endtime))
            except Exception as e:
                pass

        return (str_stdout, str_stderr, stdout, stderr, latest_lines_err_list)

    if subprocess._mswindows:
        def _readerthread(
                self, fh, buffer, is_err, process_obj,
                data_arrived_callback, error_processor, should_write_log):
            while True:
                try:
                    data_buff = fh.readline()
                    if not data_buff:
                        break
                    data_line = self._translate_newlines(data_buff,
                                                         self.stderr.encoding if is_err else self.stdout.encoding,
                                                         self.stderr.errors if is_err else self.stdout.errors) \
                        if isinstance(data_buff, bytes) else data_buff.replace("\n", "")

                except EOFError as e:
                    break

                if data_line is None:
                    break
                if data_arrived_callback is not None:
                    data_arrived_callback([data_line], is_err=is_err,
                                          process_obj=process_obj, error_processor=error_processor,
                                          should_write_log=should_write_log)
                buffer.append(data_line)
            fh.close()

        def _communicate(self, input, endtime, orig_timeout,
                         data_arrived_callback=None, error_processor=None, should_write_log=True):
            # Start reader threads feeding into a list hanging off of this
            # object, unless they've already been started.
            if self.stdout and not hasattr(self, "_stdout_buff"):
                self._stdout_buff = []
                self.stdout_thread = \
                    threading.Thread(target=self._readerthread,
                                     args=(self.stdout, self._stdout_buff,
                                           False, self,
                                           data_arrived_callback, error_processor,
                                           should_write_log))
                self.stdout_thread.daemon = True
                self.stdout_thread.start()
            if self.stderr and not hasattr(self, "_stderr_buff"):
                self._stderr_buff = []
                self.stderr_thread = \
                    threading.Thread(target=self._readerthread,
                                     args=(self.stderr, self._stderr_buff,
                                           True, self,
                                           data_arrived_callback, error_processor,
                                           should_write_log))
                self.stderr_thread.daemon = True
                self.stderr_thread.start()

            if self.stdin:
                self._stdin_write(input)

            # Wait for the reader threads, or time out.  If we time out, the
            # threads remain reading and the fds left open in case the user
            # calls communicate again.
            if self.stdout is not None:
                self.stdout_thread.join(self._remaining_time(endtime))
                if self.stdout_thread.is_alive():
                    pass
                    # raise TimeoutExpired(self.args, orig_timeout)
            if self.stderr is not None:
                self.stderr_thread.join(self._remaining_time(endtime))
                if self.stderr_thread.is_alive():
                    pass
                    # raise TimeoutExpired(self.args, orig_timeout)

            # Collect the output from and close both pipes, now that we know
            # both have been read successfully.
            stdout = None
            stderr = None
            if self.stdout:
                stdout = self._stdout_buff
                self.stdout.close()
            if self.stderr:
                stderr = self._stderr_buff
                self.stderr.close()

            # All data exchanged.  Translate lists into strings.
            # stdout = stdout[0] if stdout else None
            # stderr = stderr[0] if stderr else None
            str_stdout = None
            str_stderr = None
            latest_lines_err_list = list()
            if stdout is not None:
                str_stdout = ''.join(stdout)
            if stderr is not None:
                str_stderr = ''.join(stderr)

                min_len = min(len(stderr), Popen.LATEST_ERROR_LINE_NUM)
                for err_line_index in range(-min_len, 0):
                    latest_lines_err_list.append(stderr[err_line_index])

            return (str_stdout, str_stderr, stdout, stderr, latest_lines_err_list)
    else:
        def _communicate(self, input, endtime, orig_timeout,
                         data_arrived_callback=None, error_processor=None, should_write_log=True):
            if self.stdin and not self._communication_started:
                # Flush stdio buffer.  This might block, if the user has
                # been writing to .stdin in an uncontrolled fashion.
                try:
                    self.stdin.flush()
                except BrokenPipeError:
                    pass  # communicate() must ignore BrokenPipeError.
                if not input:
                    try:
                        self.stdin.close()
                    except BrokenPipeError:
                        pass  # communicate() must ignore BrokenPipeError.

            stdout = None
            stderr = None

            # Only create this mapping if we haven't already.
            if not self._communication_started:
                self._fileobj2output = {}
                if self.stdout:
                    self._fileobj2output[self.stdout] = []
                if self.stderr:
                    self._fileobj2output[self.stderr] = []

            if self.stdout:
                stdout = self._fileobj2output[self.stdout]
            if self.stderr:
                stderr = self._fileobj2output[self.stderr]

            self._save_input(input)

            if self._input:
                input_view = memoryview(self._input)

            prev_stdout_data_lines = None
            prev_stderr_data_lines = None
            with subprocess._PopenSelector() as selector:
                if self.stdin and input:
                    selector.register(self.stdin, selectors.EVENT_WRITE)
                if self.stdout and not self.stdout.closed:
                    selector.register(self.stdout, selectors.EVENT_READ)
                if self.stderr and not self.stderr.closed:
                    selector.register(self.stderr, selectors.EVENT_READ)

                while selector.get_map():
                    timeout = self._remaining_time(endtime)
                    if timeout is not None and timeout < 0:
                        try:
                            self._check_timeout(endtime, orig_timeout,
                                                stdout, stderr,
                                                skip_check_and_raise=True)
                            raise RuntimeError(  # Impossible :)
                                '_check_timeout(..., skip_check_and_raise=True) '
                                'failed to raise TimeoutExpired.')
                        except Exception as e:
                            pass

                    ready = selector.select(timeout)
                    try:
                        self._check_timeout(endtime, orig_timeout, stdout, stderr)
                    except Exception as e:
                        pass

                    # XXX Rewrite these to use non-blocking I/O on the file
                    # objects; they are no longer using C stdio!

                    for key, events in ready:
                        if key.fileobj is self.stdin:
                            chunk = input_view[self._input_offset:
                                               self._input_offset + self.TimeoutExpired._PIPE_BUF]
                            try:
                                self._input_offset += os.write(key.fd, chunk)
                            except BrokenPipeError:
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                            else:
                                if self._input_offset >= len(self._input):
                                    selector.unregister(key.fileobj)
                                    key.fileobj.close()
                        elif key.fileobj in (self.stdout, self.stderr):
                            data = os.read(key.fd, 32768)
                            if not data:
                                selector.unregister(key.fileobj)
                                key.fileobj.close()
                            self._fileobj2output[key.fileobj].append(data)

                            data_str = b''.join(self._fileobj2output[key.fileobj])
                            if self.text_mode:
                                if key.fileobj == self.stdout:
                                    data_str = self._translate_newlines(data_str,
                                                                        self.stdout.encoding,
                                                                        self.stdout.errors)
                                else:
                                    data_str = self._translate_newlines(data_str,
                                                                        self.stderr.encoding,
                                                                        self.stderr.errors)

                            if len(data_str) >= 1:
                                diff_data_lines = list()
                                data_lines = data_str.splitlines()
                                if data_str[len(data_str) - 1] != '\n':
                                    data_lines.pop(len(data_lines) - 1)

                                if key.fileobj == self.stdout:
                                    if prev_stdout_data_lines is None:
                                        prev_stdout_data_lines = data_lines
                                        diff_data_lines = data_lines
                                    else:
                                        for index in range(len(prev_stdout_data_lines), len(data_lines)):
                                            diff_data_lines.append(data_lines[index])
                                        prev_stdout_data_lines = data_lines
                                else:
                                    if prev_stderr_data_lines is None:
                                        prev_stderr_data_lines = data_lines
                                        diff_data_lines = data_lines
                                    else:
                                        for index in range(len(prev_stderr_data_lines), len(data_lines)):
                                            diff_data_lines.append(data_lines[index])
                                        prev_stderr_data_lines = data_lines

                                if data_arrived_callback is not None:
                                    data_arrived_callback(
                                        diff_data_lines, is_err=True if key.fileobj == self.stderr else False,
                                        process_obj=self, error_processor=error_processor,
                                        should_write_log=should_write_log
                                    )

            try:
                self.wait(timeout=self._remaining_time(endtime))
            except Exception as e:
                pass

            # All data exchanged.  Translate lists into strings.
            str_stdout = None
            str_stderr = None
            if stdout is not None:
                str_stdout = b''.join(stdout)
            if stderr is not None:
                str_stderr = b''.join(stderr)

            latest_lines_err_list = list()
            if stderr is not None:
                min_len = min(len(stderr), Popen.LATEST_ERROR_LINE_NUM)
                for err_line_index in range(-min_len, 0):
                    if self.text_mode:
                        err_line = self._translate_newlines(stderr[err_line_index],
                                                            self.stderr.encoding,
                                                            self.stderr.errors)
                    else:
                        err_line = stderr[err_line_index]
                    latest_lines_err_list.append(err_line)

            # Translate newlines, if requested.
            # This also turns bytes into strings.
            if self.text_mode:
                if stdout is not None:
                    str_stdout = self._translate_newlines(str_stdout,
                                                          self.stdout.encoding,
                                                          self.stdout.errors)
                if stderr is not None:
                    str_stderr = self._translate_newlines(str_stderr,
                                                          self.stderr.encoding,
                                                          self.stderr.errors)

            return (str_stdout, str_stderr, stdout, stderr, latest_lines_err_list)
