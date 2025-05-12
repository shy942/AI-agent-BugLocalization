<?php

namespace LaravelEnso\Tables\Services\Data\Computors;

use Illuminate\Database\Eloquent\Collection;
use Illuminate\Database\Eloquent\Model;
use Illuminate\Support\Facades\App;
use LaravelEnso\Helpers\Services\Obj;
use LaravelEnso\Tables\Contracts\ComputesModelColumns;
use stdClass;

class Resource implements ComputesModelColumns
{
    private static Obj $columns;

    public static function columns($columns): void
    {
        self::$columns = $columns
            ->filter(fn ($column) => $column->get('resource'))
            ->values();
    }

    public static function handle(Model $row)
    {
        foreach (self::$columns as $column) {
            $value = $row->{$column->get('name')};
            $resource = $column->get('resource');
            unset($row->{$column->get('name')});

            $row->{$column->get('name')} = $value instanceof Collection
                ? self::collection($value, $resource)
                : self::resource($value, $resource);
        }

        return $row;
    }

    private static function collection($value, $resource)
    {
        return $value->isEmpty()
            ? $value
            : App::make($resource, [
                'resource' => new stdClass(),
            ])::collection($value);
    }

    private static function resource($value, $resource)
    {
        return $value === null
            ? null
            : App::make($resource, ['resource' => $value]);
    }
}
