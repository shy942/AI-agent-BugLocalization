<?php

namespace LaravelEnso\Tables\Traits;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\App;
use LaravelEnso\Tables\Services\Data\Config;
use LaravelEnso\Tables\Services\TemplateLoader;

trait ProvidesData
{
    use ProvidesRequest;

    public function data(Request $request)
    {
        $tableClass = method_exists($this, 'tableClass')
            ? $this->tableClass($request)
            : $this->tableClass;

        $request = $this->request($request);
        $table = App::make($tableClass, ['request' => $request]);
        $template = (new TemplateLoader($table))->handle();
        $config = new Config($request, $template);

        return ['table' => $table, 'config' => $config];
    }
}
